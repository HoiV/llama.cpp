/* djsample.c - DJGPP front end, sound support
 *	Original BCsample.c Copyright (c) 1995-1997 Stefan Jokisch
 *	DJGPP prot by Jim Dunleavy <jim.dunleavy@erha.ie>
 *
 * This file is part of Frotz.
 *
 * Frotz is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Frotz is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dos.h>

#ifdef SOUND_SUPPORT
#include <pc.h>
#include <dpmi.h>
#include <go32.h>
#endif

#include "frotz.h"
#include "djfrotz.h"

#ifdef SOUND_SUPPORT

#define SWAP_BYTES(v)	v = v << 8 | v >> 8;

#define READ_DSP(v)	{while(!inportb(sound_adr+14)&0x80);v=inportb(sound_adr+10);}
#define WRITE_DSP(v)	{while(inportb(sound_adr+12)&0x80);outportb(sound_adr+12,v);}

extern void end_of_sound (void);

extern FILE *os_path_open (const char *, const char *);

void end_of_dma (void);
void end_of_dma_end (void);

static struct {
    word prefix __attribute__((packed));
    byte repeats __attribute__((packed));
    byte base_note __attribute__((packed));
    word frequency __attribute__((packed));
    word unused __attribute__((packed));
    word length __attribute__((packed));
} sheader;

static current_sample = 0;

static __dpmi_paddr old_vector;

static struct {
    int play_part_ __attribute__((packed));
    int play_count_ __attribute__((packed));
    long sample_adr1_ __attribute__((packed));
    long sample_adr2_ __attribute__((packed));
    int end_of_sound_flag_ __attribute__((packed));
    word sample_len1_ __attribute__((packed));
    word sample_len2_ __attribute__((packed));
    word sound_adr_ __attribute__((packed));
    word sound_irq_ __attribute__((packed));
    word sound_dma_ __attribute__((packed));
    word dma_page_port_[4] __attribute__((packed));
} nonpaged_data = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    { 0x87, 0x83, 0x81, 0x82 }
};

#define play_part nonpaged_data.play_part_
#define play_count nonpaged_data.play_count_
#define sound_adr nonpaged_data.sound_adr_
#define sound_irq nonpaged_data.sound_irq_
#define sound_dma nonpaged_data.sound_dma_
#define sample_adr1 nonpaged_data.sample_adr1_
#define sample_adr2 nonpaged_data.sample_adr2_
#define sample_len1 nonpaged_data.sample_len1_
#define sample_len2 nonpaged_data.sample_len2_
#define dma_page_port nonpaged_data.dma_page_port_

static word sound_int = 0;
static word sound_ver = 0;

static byte *sample_data = NULL;

static int sample_sel;
static int sample_seg = -1;

/*
 * start_of_dma
 *
 * Start the DMA transfer to the sound board.
 *
 */

static void start_of_dma (long address, unsigned length)
{

    length--;

    /* Set up DMA chip */

    outportb (0x0a, 0x04 | sound_dma);
    outportb (0x0c, 0x00);
    outportb (0x0b, 0x48 | sound_dma);
    outportb (2 * sound_dma, byte0 (address));
    outportb (2 * sound_dma, byte1 (address));
    outportb (dma_page_port[sound_dma], byte2 (address));
    outportb (2 * sound_dma + 1, byte0 (length));
    outportb (2 * sound_dma + 1, byte1 (length));
    outportb (0x0a, sound_dma);

    /* Play 8-bit mono sample */

    WRITE_DSP (0x14)
    WRITE_DSP (byte0 (length))
    WRITE_DSP (byte1 (length))

}/* start_of_dma */

/*
 * end_of_dma
 *
 * This function is called when a hardware interrupt signals the
 * end of the current sound. We may have to play the second half
 * of the sound effect, or we may have to repeat it, or call the
 * end_of_sound function when we are finished.
 *
 */

asm ("

_play_part = _nonpaged_data
_play_count = _nonpaged_data + 4
_sample_adr1 = _nonpaged_data + 8
_sample_adr2 = _nonpaged_data + 12
_end_of_sound_flag = _nonpaged_data + 16
_sample_len1 = _nonpaged_data + 20
_sample_len2 = _nonpaged_data + 22
_sound_adr = _nonpaged_data + 24
_sound_irq = _nonpaged_data + 26
_sound_dma = _nonpaged_data + 28
	.globl _end_of_sound_flag

	.globl _end_of_dma
	.align 4
_end_of_dma:
	pushw %ds		/* save register */
	pushal

	movw %cs:___djgpp_ds_alias, %ax
	movw %ax, %ds		/* set up selector */
	
	/* Play the second half,play another cycle or finish */
	cmpl $1,_play_part
	jne check_repeats
	cmpw $0,_sample_len2
	je check_repeats
	movl $2,_play_part
	movzwl _sample_len2,%eax
	pushl %eax
	pushl _sample_adr2
	jmp restart_dma
	.align 2,0x90
check_repeats:
	movl _play_count,%edx
	cmpl $255,%edx
	je play_first_part
	decl %edx
	movl %edx,_play_count
	je set_eos
play_first_part:
	movl $1,_play_part
	movzwl _sample_len1,%eax
	pushl %eax
	pushl _sample_adr1
restart_dma:
	call _start_of_dma
	addl $8,%esp
	jmp eoi
	.align 2,0x90
set_eos:
	movl %edx,_play_part	/* Clear play_part */
	movl $1,_end_of_sound_flag
eoi:
	/* Tell interrupt controller(s) + sound board we are done */
	movb $0x20,%al
	outb %al,$0x20
	cmpw $7,_sound_irq
	jbe 1f
	outb %al,$0xa0
1:
	movw _sound_adr,%dx
	addl $14,%edx
	inb %dx,%al

	popal			/* iret */
	popw %ds
	sti
	iret


	.globl _end_of_dma_end
	.align 4
_end_of_dma_end:
");

/*
 * init_sound
 *
 * Initialise the sound board and various sound related variables.
 *
 */

bool init_sound (void)
{
    const char *settings;
    word irc_mask_port;
    __dpmi_paddr addr;

    /* Read the IRQ, port address, DMA channel and SB version */

    if ((settings = getenv ("BLASTER")) == NULL)
	return FALSE;

    sound_irq = dectoi (strchr (settings, 'I') + 1);
    sound_adr = hextoi (strchr (settings, 'A') + 1);
    sound_dma = dectoi (strchr (settings, 'D') + 1);
    sound_ver = dectoi (strchr (settings, 'T') + 1);

    /* Reset mixer chip and DSP */

    outportb (sound_adr + 4, 0);
    outportb (sound_adr + 5, 0);

    outportb (sound_adr + 6, 1);
    inportb (sound_adr + 6);
    inportb (sound_adr + 6);
    inportb (sound_adr + 6);
    outportb (sound_adr + 6, 0);

    /* Turn on speakers */

    WRITE_DSP (0xd1)

    /* Install the end_of_dma interrupt */

    if (sound_irq < 8) {
	irc_mask_port = 0x21;
	sound_int = 0x08 + sound_irq;
    } else {
	irc_mask_port = 0xa1;
	sound_int = 0x68 + sound_irq;
    }

    addr.selector = _my_cs();
    addr.offset32 = (long)end_of_dma;
    __dpmi_get_protected_mode_interrupt_vector (sound_int, &old_vector);
    __dpmi_set_protected_mode_interrupt_vector (sound_int, &addr);

    /* Allocate 64KB for sample data */

    if ((sample_data = malloc (0x10000L)) == NULL)
	return FALSE;

    /* allocate DOS memory for DMA */
    sample_seg = __dpmi_allocate_dos_memory (0x1000, &sample_sel);

    if (sample_seg < 0)
       return FALSE;

    sample_adr1 = sample_seg << 4;
    sample_adr2 = sample_adr1;
    word0 (sample_adr2) = 0;
    ++word1 (sample_adr2);

    /* Lock code and data used by the end_of_dma interrupt handler */

    _go32_dpmi_lock_code (start_of_dma,
			  (long)end_of_dma_end - (long)start_of_dma);
    _go32_dpmi_lock_data ((void *)&nonpaged_data, sizeof(nonpaged_data));

    /* Enable the end_of_dma interrupt */

    outportb (0x20, 0x20);

    if (sound_irq >= 8)
	outportb (0xa0, 0x20);

    outportb (irc_mask_port, inportb (irc_mask_port) & ~(1 << (sound_irq & 7)));

    /* Indicate success */

    return TRUE;

}/* init_sound */

/*
 * reset_sound
 *
 * Free resources allocated for playing samples.
 *
 */

void reset_sound (void)
{

    os_stop_sample ();

    if (sample_data != NULL) {
	free (sample_data);
	sample_data = NULL;
    }
    if (sample_seg > 0) {
	__dpmi_free_dos_memory (sample_sel);
	sample_seg = 0;
    }
    if (sound_adr != 0) {
	__dpmi_set_protected_mode_interrupt_vector (sound_int, &old_vector);
	sound_adr = 0;
    }

}/* reset_sound */

#endif

/*
 * os_beep
 *
 * Play a beep sound. Ideally, the sound should be high- (number == 1)
 * or low-pitched (number == 2).
 *
 */

void os_beep (int number)
{
    word T = 888 * number;

    outportb (0x43, 0xb6);
    outportb (0x42, lo (T));
    outportb (0x42, hi (T));
    outportb (0x61, inportb (0x61) | 3);

    delay (75);

    outportb (0x61, inportb (0x61) & ~3);

}/* os_beep */

/*
 * os_prepare_sample
 *
 * Load the sample from the disk.
 *
 */

void os_prepare_sample (int number)
{
#ifdef SOUND_SUPPORT

    os_stop_sample ();

    /* Exit if the sound board isn't set up properly */

    if (sample_data == NULL)
	return;
    if (sound_adr == 0)
	return;

    /* Continue only if the desired sample is not already present */

    if (current_sample != number) {

	char sample_name[MAX_FILE_NAME + 1];
	char numstr[2];
	FILE *fp;

	/* Build sample file name */

	strcpy (sample_name, "sound\\");

	numstr[0] = '0' + number / 10;
	numstr[1] = '0' + number % 10;

	strncat (sample_name, stripped_story_name, 6);
	strncat (sample_name, numstr, 2);
	strncat (sample_name, ".snd", 4);

	/* Open sample file */

	if ((fp = os_path_open (sample_name, "rb")) == NULL)
	    return;

	/* Load header and sample data */

	fread (&sheader, sizeof (sheader), 1, fp);

	SWAP_BYTES (sheader.frequency)
	SWAP_BYTES (sheader.length)

	fread (sample_data, 1, sheader.length, fp);
	dosmemput (sample_data, sheader.length, sample_adr1);

	sample_len1 = -(word) sample_adr1;

	if (sample_len1 > sheader.length || sample_len1 == 0)
	    sample_len1 = sheader.length;

	sample_len2 = sheader.length - sample_len1;

	WRITE_DSP (0x40)
	WRITE_DSP (256 - 1000000L / sheader.frequency)

	current_sample = number;

	/* Close sample file */

	fclose (fp);

    }

#endif
}/* os_prepare_sample */

/*
 * os_start_sample
 *
 * Play the given sample at the given volume (ranging from 1 to 8 and
 * 255 meaning a default volume). The sound is played once or several
 * times in the background (255 meaning forever). The end_of_sound
 * function is called as soon as the sound finishes.
 *
 */

void os_start_sample (int number, int volume, int repeats)
{
#ifdef SOUND_SUPPORT

    os_stop_sample ();

    /* Exit if the sound board isn't set up properly */

    if (sample_data == NULL)
	return;
    if (sound_adr == 0)
	return;

    /* Load new sample */

    os_prepare_sample (number);

    /* Continue only if the sample's in memory now */

    if (current_sample == number) {

	play_count = repeats;

	if (sound_ver < 6) {	/* Set up SB pro mixer chip */

	    volume = (volume != 255) ? 7 + volume : 15;

	    outportb (sound_adr + 4, 0x04);
	    outportb (sound_adr + 5, (volume << 4) | volume);
	    outportb (sound_adr + 4, 0x22);
	    outportb (sound_adr + 5, 0xff);

	} else {		/* Set up SB16 mixer chip */

	    /* Many thanks to Linards Ticmanis for writing this part! */

	    volume = (volume != 255) ? 127 + 16 * volume : 255;

	    outportb (sound_adr + 4, 0x32);
	    outportb (sound_adr + 5, volume);
	    outportb (sound_adr + 4, 0x33);
	    outportb (sound_adr + 5, volume);
	    outportb (sound_adr + 4, 0x30);
	    outportb (sound_adr + 5, 0xff);
	    outportb (sound_adr + 4, 0x31);
	    outportb (sound_adr + 5, 0xff);

	}

	play_part = 1;
	start_of_dma (sample_adr1, sample_len1);

    }

#endif
}/* os_start_sample */

/*
 * os_stop_sample
 *
 * Turn off the current sample.
 *
 */

void os_stop_sample (void)
{
#ifdef SOUND_SUPPORT

    play_part = 0;

    /* Exit if the sound board isn't set up properly */

    if (sample_data == NULL)
	return;
    if (sound_adr == 0)
	return;

    /* Tell DSP to stop the current sample */

    WRITE_DSP (0xd0)

#endif
}/* os_stop_sample */

/*
 * os_finish_with_sample
 *
 * Remove the current sample from memory (if any).
 *
 */

void os_finish_with_sample (void)
{
#ifdef SOUND_SUPPORT

    os_stop_sample ();		/* we keep 64KB allocated all the time */

#endif
}/* os_finish_with_sample */
