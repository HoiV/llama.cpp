/* djinit.c - DJGPP front end, initialisation
 *	Original BCinit.c Copyright (c) 1995-1997 Stefan Jokisch
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

#include <signal.h>
#include <pc.h>
#include <go32.h>
#include <dpmi.h>
#include <conio.h>
#include <dos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "frotz.h"
#include "djfrotz.h"

static char information[] =
"\n"
"FROTZ V2.40 - interpreter for all Infocom games. Complies with standard\n"
"1.0 of Graham Nelson's specification. Written by Stefan Jokisch in 1995-7\n"
"\n"
"Syntax: frotz [options] story-file\n"
"\n"
"  -a   watch attribute setting  \t -o   watch object movement\n"
"  -A   watch attribute testing  \t -O   watch object locating\n"
"  -b # background colour        \t -p   alter piracy opcode\n"
"  -B # reverse background colour\t -r # right margin\n"
"  -c # context lines            \t -R   save/restore in old Frotz format\n"
"  -d # display mode (see below) \t -s # random number seed value\n"
"  -e # emphasis colour [mode 1] \t -S # transscript width\n"
"  -f # foreground colour        \t -t   set Tandy bit\n"
"  -F # reverse foreground colour\t -T   bold typing [modes 2+4+5]\n"
"  -g # font [mode 5] (see below)\t -u # slots for multiple undo\n"
"  -h # screen height            \t -w # screen width\n"
"  -i   ignore runtime errors    \t -x   expand abbreviations g/x/z\n"
"  -l # left margin"
"              \t -Z # error checking (see below)"
"\n\n"
"Fonts are 0 (fixed), 1 (sans serif), 2 (comic), 3 (times), 4 (serif).\n"
"\n"
"Display modes are 0 (mono), 1 (text), 2 (CGA), 3 (MCGA), 4 (EGA), 5 (Amiga)."
"\n\n"
"Error checking is 0 (none), 1 (report first error (default)),\n"
"  2 (report all errors), 3 (exit after any error).";



extern const char *optarg;
extern int optind;

int getopt (int, char *[], const char *);

static const char *progname = NULL;

extern char script_name[];
extern char command_name[];
extern char save_name[];
extern char auxilary_name[];

char stripped_story_name[10];

int display = -1;

int user_background = -1;
int user_foreground = -1;
int user_emphasis = -1;
int user_bold_typing = -1;
int user_reverse_bg = -1;
int user_reverse_fg = -1;
int user_screen_height = -1;
int user_screen_width = -1;
int user_tandy_bit = -1;
int user_random_seed = -1;
int user_font = 1;

static byte old_video_mode = 0;


/*
 * dectoi
 *
 * Convert a string containing a decimal number to integer. The string may
 * be NULL, but it must not be empty.
 *
 */

int dectoi (const char *s)
{
    int n = 0;

    if (s != NULL)

	do {

	    n = 10 * n + (*s & 15);

	} while (*++s > ' ');

    return n;

}/* dectoi */

/*
 * hextoi
 *
 * Convert a string containing a hex number to integer. The string may be
 * NULL, but it must not be empty.
 *
 */

int hextoi (const char *s)
{
    int n = 0;

    if (s != NULL)

	do {

	    n = 16 * n + (*s & 15);

	    if (*s > '9')
		n += 9;

	} while (*++s > ' ');

    return n;

}/* hextoi */

/*
 * cleanup
 *
 * Shut down the IO interface: free memory, close files, restore
 * interrupt pointers and return to the previous video mode.
 *
 */

static void cleanup (void)
{
    __dpmi_regs regs;

#ifdef SOUND_SUPPORT
    reset_sound ();
#endif
    reset_pictures ();

    regs.h.ah = 0;
    regs.h.al = old_video_mode;
    __dpmi_int (0x10, &regs);

    signal(SIGINT, SIG_DFL);

}/* cleanup */

/*
 * fast_exit
 *
 * Handler routine to be called when the crtl-break key is pressed.
 *
 */

static void fast_exit ()
{

    cleanup (); exit (EXIT_FAILURE);

}/* fast_exit */

/*
 * os_fatal
 *
 * Display error message and exit program.
 *
 */

void os_fatal (const char *s)
{

    if (h_interpreter_number)
	os_reset_screen ();

    /* Display error message */

    fputs ("\nFatal error: ", stderr);
    fputs (s, stderr);
    fputs ("\n", stderr);

    /* Abort program */

    exit (EXIT_FAILURE);

}/* os_fatal */

/*
 * parse_options
 *
 * Parse program options and set global flags accordingly.
 *
 */

static void parse_options (int argc, char **argv)
{
    int c;

    do {

	int num = 0;

	c = getopt (argc, argv, "aAb:B:c:d:e:f:F:g:h:il:oOpr:Rs:S:tTu:w:xZ:");

	if (optarg != NULL)
	    num = dectoi (optarg);

	if (c == 'a')
	    option_attribute_assignment = 1;
	if (c == 'A')
	    option_attribute_testing = 1;
	if (c == 'b')
	    user_background = num;
	if (c == 'B')
	    user_reverse_bg = num;
	if (c == 'c')
	    option_context_lines = num;
	if (c == 'd') {
	    display = optarg[0] | 32;
	    if ((display < '0' || display > '5')
		&& (display < 'a' || display > 'e')) {
		display = -1;
	    }
	}
	if (c == 'e')
	    user_emphasis = num;
	if (c == 'T')
	    user_bold_typing = 1;
	if (c == 'f')
	    user_foreground = num;
	if (c == 'F')
	    user_reverse_fg = num;
	if (c == 'g')
	    if (num >= 0 && num <= 4)
		user_font = num;
	if (c == 'h')
	    user_screen_height = num;
	if (c == 'i')
	    option_ignore_errors = 1;
	if (c == 'l')
	    option_left_margin = num;
	if (c == 'o')
	    option_object_movement = 1;
	if (c == 'O')
	    option_object_locating = 1;
	if (c == 'p')
	    option_piracy = 1;
	if (c == 'r')
	    option_right_margin = num;
	if (c == 'R')
	    option_save_quetzal = 0;
	if (c == 's')
	    user_random_seed = num;
	if (c == 'S')
	    option_script_cols = num;
	if (c == 't')
	    user_tandy_bit = 1;
	if (c == 'u')
	    option_undo_slots = num;
	if (c == 'w')
	    user_screen_width = num;
	if (c == 'x')
	    option_expand_abbreviations = 1;
	if (c == 'Z')
	    if (num >= ERR_REPORT_NEVER && num <= ERR_REPORT_FATAL)
	      err_report_mode = num;
    if (c == '?')
	optind = argc;

    } while (c != EOF && c != '?');

}/* parse_options */

/*
 * os_process_arguments
 *
 * Handle command line switches. Some variables may be set to activate
 * special features of Frotz:
 *
 *     option_attribute_assignment
 *     option_attribute_testing
 *     option_context_lines
 *     option_object_locating
 *     option_object_movement
 *     option_left_margin
 *     option_right_margin
 *     option_ignore_errors
 *     option_piracy
 *     option_undo_slots
 *     option_expand_abbreviations
 *     option_script_cols
 *
 * The global pointer "story_name" is set to the story file name.
 *
 */

void os_process_arguments (int argc, char *argv[])
{
    const char *p;
    int i;

    /* Parse command line options */

    parse_options (argc, argv);

    if (optind != argc - 1) {
	puts (information);
	exit (EXIT_FAILURE);
    }

    /* Set the story file name */

    story_name = argv[optind];

    /* Strip path and extension off the story file name */

    p = story_name;

    for (i = 0; story_name[i] != 0; i++)
	if (story_name[i] == '\\' || story_name[i] == '/'
	    || story_name[i] == ':')
	    p = story_name + i + 1;

    for (i = 0; p[i] != 0 && p[i] != '.'; i++)
	stripped_story_name[i] = p[i];

    stripped_story_name[i] = 0;

    /* Create nice default file names */

    strcpy (script_name, stripped_story_name);
    strcpy (command_name, stripped_story_name);
    strcpy (save_name, stripped_story_name);
    strcpy (auxilary_name, stripped_story_name);

    strcat (script_name, ".scr");
    strcat (command_name, ".rec");
    strcat (save_name, ".sav");
    strcat (auxilary_name, ".aux");

    /* Save the executable file name */

    progname = argv[0];

}/* os_process_arguments */

/*
 * set_palette
 *
 * Set palette registers and call VGA BIOS to use DAC registers.
 * If special is zero, set registers to EGA default values and
 * use DAC registers #0 to #63.
 * If special is non-zero, set palette register #i to value i
 * and use DAC registers #64 to #127.
 *
 */

static void set_palette (int special)
{

    static byte standard_palette[] = {
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x14, 0x07,
	0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
	0x00 /* last one is the overscan register */
    };
    static byte special_palette[] = {
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x00 /* last one is the overscan register */
    };
    byte *p = special ? special_palette : standard_palette;
    __dpmi_regs regs;

    if (display == _AMIGA_) {
	dosmemput(p, sizeof(standard_palette), MASK_LINEAR(__tb));
	regs.x.ax = 0x1002;
	regs.x.dx = RM_OFFSET(__tb);
	regs.x.es = RM_SEGMENT(__tb);
	__dpmi_int (0x10, &regs);
	regs.x.ax = 0x1013;
	regs.x.bx = special ? 0x0101 : 0x0001;
	__dpmi_int (0x10, &regs);
    }

}/* set_palette */

/*
 * os_init_screen
 *
 * Initialise the IO interface. Prepare the screen and other devices
 * (mouse, sound board). Set various OS depending story file header
 * entries:
 *
 *     h_config (aka flags 1)
 *     h_flags (aka flags 2)
 *     h_screen_cols (aka screen width in characters)
 *     h_screen_rows (aka screen height in lines)
 *     h_screen_width
 *     h_screen_height
 *     h_font_height (defaults to 1)
 *     h_font_width (defaults to 1)
 *     h_default_foreground
 *     h_default_background
 *     h_interpreter_number
 *     h_interpreter_version
 *     h_user_name (optional; not used by any game)
 *
 * Finally, set reserve_mem to the amount of memory (in bytes) that
 * should not be used for multiple undo and reserved for later use.
 *
 */

void os_init_screen (void)
{
    static byte zcolour[] = {
	BLACK_COLOUR,
	BLUE_COLOUR,
	GREEN_COLOUR,
	CYAN_COLOUR,
	RED_COLOUR,
	MAGENTA_COLOUR,
	BROWN + 16,
	LIGHTGRAY + 16,
	GREY_COLOUR,
	LIGHTBLUE + 16,
	LIGHTGREEN + 16,
	LIGHTCYAN + 16,
	LIGHTRED + 16,
	LIGHTMAGENTA + 16,
	YELLOW_COLOUR,
	WHITE_COLOUR
    };

    static struct {	/* information on modes 0 to 5 */
	byte vmode;
	word width;
	word height;
	byte font_width;
	byte font_height;
	byte fg;
	byte bg;
    } info[] = {
	{ 0x07,  80,  25,  1,  1, LIGHTGRAY + 16, BLACK_COLOUR }, /* MONO  */
	{ 0x03,  80,  25,  1,  1, LIGHTGRAY + 16, BLUE_COLOUR  }, /* TEXT  */
	{ 0x06, 640, 200,  8,  8, WHITE_COLOUR,   BLACK_COLOUR }, /* CGA   */
	{ 0x13, 320, 200,  5,  8, WHITE_COLOUR,   GREY_COLOUR  }, /* MCGA  */
	{ 0x0e, 640, 200,  8,  8, WHITE_COLOUR,   BLUE_COLOUR  }, /* EGA   */
	{ 0x12, 640, 400,  8, 16, WHITE_COLOUR,   BLACK_COLOUR }  /* AMIGA */
    };

    static struct {	/* information on modes A to E */
	word vesamode;
	word width;
	word height;
    } subinfo[] = {
	{ 0x001,  40, 25 },
	{ 0x109, 132, 25 },
	{ 0x10b, 132, 50 },
	{ 0x108,  80, 60 },
	{ 0x10c, 132, 60 }
    };

    int subdisplay;
    __dpmi_regs regs;

    /* Get the current video mode. This video mode will be selected
       when the program terminates. It's also useful to auto-detect
       monochrome boards. */

    regs.h.ah = 15;
    __dpmi_int (0x10, &regs);
    old_video_mode = regs.h.al;

    /* If the display mode has not already been set by the user then see
       if this is a monochrome board. If so, set the display mode to 0.
       Otherwise check the graphics flag of the story. Select a graphic
       mode if it is set or if this is a V6 game. Select text mode if it
       is not. */

    if (display == -1)

	if (old_video_mode == 7)
	    display = '0';
	else if (h_version == V6 || (h_flags & GRAPHICS_FLAG))
	    display = '5';
	else
	    display = '1';

    /* Activate the desired display mode. All VESA text modes are very
       similar to the standard text mode; in fact, only here we need to
       know which VESA mode is used. */

    if (display >= '0' && display <= '5') {
	subdisplay = -1;
	display -= '0';
	regs.h.al = info[display].vmode;
	regs.h.ah = 0;
    } else if (display == 'a') {
	subdisplay = 0;
	display = 1;
	regs.x.ax = 0x0001;
    } else {
	subdisplay = display - 'a';
	display = 1;
	regs.x.bx = subinfo[subdisplay].vesamode;
	regs.x.ax = 0x4f02;
    }

    __dpmi_int (0x10, &regs);

    /* Make various preparations */

    if (display <= _TEXT_) {

	/* Enable bright background colours */

	regs.x.ax = 0x1003;
	regs.h.bl = 0;
	__dpmi_int (0x10, &regs);

	/* Turn off hardware cursor */

	regs.h.ah = 1;
	regs.x.cx = 0xffff;
	__dpmi_int (0x10, &regs);

    } else {

	load_fonts ();

	if (display == _AMIGA_) {

	     scaler = 2;

	     /* Use resolution 640 x 400 instead of 640 x 480. BIOS doesn't
		help us here since this is not a standard resolution. */

	     outportb (0x03c2, 0x63);

	     outportw (0x03d4, 0x0e11);
	     outportw (0x03d4, 0xbf06);
	     outportw (0x03d4, 0x1f07);
	     outportw (0x03d4, 0x9c10);
	     outportw (0x03d4, 0x8f12);
	     outportw (0x03d4, 0x9615);
	     outportw (0x03d4, 0xb916);

	 }

    }

    /* Amiga emulation under V6 needs special preparation. */

    if (display == _AMIGA_ && h_version == V6) {

	user_reverse_fg = -1;
	user_reverse_bg = -1;
	zcolour[LIGHTGRAY] = LIGHTGREY_COLOUR;
	zcolour[DARKGRAY] = DARKGREY_COLOUR;

	set_palette (1);

    }

    /* Set various bits in the configuration byte. These bits tell
       the game which features are supported by the interpreter. */

    if (h_version == V3 && user_tandy_bit != -1)
	h_config |= CONFIG_TANDY;
    if (h_version == V3)
	h_config |= CONFIG_SPLITSCREEN;
    if (h_version == V3 && (display == _MCGA_ || (display == _AMIGA_ && user_font != 0)))
	h_config |= CONFIG_PROPORTIONAL;
    if (h_version >= V4 && display != _MCGA_ && (user_bold_typing != -1 || display <= _TEXT_))
	h_config |= CONFIG_BOLDFACE;
    if (h_version >= V4)
	h_config |= CONFIG_EMPHASIS | CONFIG_FIXED | CONFIG_TIMEDINPUT;
    if (h_version >= V5 && display != _MONO_ && display != _CGA_)
	h_config |= CONFIG_COLOUR;
    if (h_version >= V5 && display >= _CGA_ && init_pictures ())
	h_config |= CONFIG_PICTURES;

    /* Handle various game flags. These flags are set if the game wants
       to use certain features. The flags must be cleared if the feature
       is not available. */

    if (h_flags & GRAPHICS_FLAG)
	if (display <= _TEXT_)
	    h_flags &= ~GRAPHICS_FLAG;
    if (h_version == V3 && (h_flags & OLD_SOUND_FLAG))
#ifdef SOUND_SUPPORT
	if (!init_sound ())
#endif
	    h_flags &= ~OLD_SOUND_FLAG;
    if (h_flags & SOUND_FLAG)
#ifdef SOUND_SUPPORT
	if (!init_sound ())
#endif
	    h_flags &= ~SOUND_FLAG;
    if (h_version >= V5 && (h_flags & UNDO_FLAG))
	if (!option_undo_slots)
	    h_flags &= ~UNDO_FLAG;
    if (h_flags & MOUSE_FLAG)
	if (subdisplay != -1 || !detect_mouse ())
	    h_flags &= ~MOUSE_FLAG;
    if (h_flags & COLOUR_FLAG)
	if (display == _MONO_ || display == _CGA_)
	    h_flags &= ~COLOUR_FLAG;
    h_flags &= ~MENU_FLAG;

    /* Set the screen dimensions, font size and default colour */

    h_screen_width = info[display].width;
    h_screen_height = info[display].height;
    h_font_height = info[display].font_height;
    h_font_width = info[display].font_width;
    h_default_foreground = info[display].fg;
    h_default_background = info[display].bg;

    if (subdisplay != -1) {
	h_screen_width = subinfo[subdisplay].width;
	h_screen_height = subinfo[subdisplay].height;
    }

    if (user_screen_width != -1)
	h_screen_width = user_screen_width;
    if (user_screen_height != -1)
	h_screen_height = user_screen_height;

    h_screen_cols = h_screen_width / h_font_width;
    h_screen_rows = h_screen_height / h_font_height;

    if (user_foreground != -1)
	h_default_foreground = zcolour[user_foreground];
    if (user_background != -1)
	h_default_background = zcolour[user_background];

    /* Set the interpreter number (a constant telling the game which
       operating system it runs on) and the interpreter version. The
       interpreter number has effect on V6 games and "Beyond Zork". */

    h_interpreter_number = INTERP_MSDOS;
    h_interpreter_version = 'F';

    if (display == _AMIGA_)
	h_interpreter_number = INTERP_AMIGA;

     /* Install the fast_exit routine to handle the ctrl-break key */

    signal(SIGINT, fast_exit);

}/* os_init_screen */

/*
 * os_reset_screen
 *
 * Reset the screen before the program stops.
 *
 */

void os_reset_screen (void)
{

    os_set_font (TEXT_FONT);
    os_set_text_style (0);
    os_display_string ((zchar *) "[Hit any key to exit.]");
    os_read_key (0, TRUE);

    cleanup ();

}/* os_reset_screen */

/*
 * os_restart_game
 *
 * This routine allows the interface to interfere with the process of
 * restarting a game at various stages:
 *
 *     RESTART_BEGIN - restart has just begun
 *     RESTART_WPROP_SET - window properties have been initialised
 *     RESTART_END - restart is complete
 *
 */

void os_restart_game (int stage)
{
    int x, y;
    __dpmi_regs regs;

    if (story_id == BEYOND_ZORK)

	if (stage == RESTART_BEGIN)

	    if ((display == _MCGA_ || display == _AMIGA_) && os_picture_data (1, &x, &y)) {

		set_palette (1);

		regs.x.ax = 0x1010;
		regs.x.bx = 64;
		regs.h.dh = 0;
		regs.x.cx = 0;
		__dpmi_int (0x10, &regs);
		regs.x.ax = 0x1010;
		regs.x.bx = 79;
		regs.h.dh = 0xff;
		regs.x.cx = 0xffff;
		__dpmi_int (0x10, &regs);

		os_draw_picture (1, 1, 1);
		os_read_key (0, FALSE);

		set_palette (0);

	    }

}/* os_restart_game */

/*
 * os_random_seed
 *
 * Return an appropriate random seed value in the range from 0 to
 * 32767, possibly by using the current system time.
 *
 */

int os_random_seed (void)
{
    __dpmi_regs regs;

    if (user_random_seed == -1) {

	/* Use the time of day as seed value */

	regs.h.ah = 0;
	__dpmi_int (0x1a, &regs);

	return regs.x.dx & 0x7fff;

    } else return user_random_seed;

}/* os_random_seed */

/*
 * os_path_open
 *
 * Open a file in the current directory.  If this fails then
 * search the directories in the ZCODE_PATH environment variable,
 * if it is defined, otherwise search INFOCOM_PATH.
 *
 */

FILE *os_path_open (const char *name, const char *mode)
{
    FILE *fp;
    char buf[MAX_FILE_NAME + 1];
    char *p, *bp, lastch;

    if ((fp = fopen (name, mode)) != NULL)
	return fp;
    if ((p = getenv ("ZCODE_PATH")) == NULL)
	p = getenv ("INFOCOM_PATH");
    if (p != NULL) {
	while (*p) {
	    bp = buf;
	    while (*p && *p != OS_PATHSEP)
		lastch = *bp++ = *p++;
	    if (lastch != '\\' && lastch != '/')
		*bp++ = '\\';
	    strcpy (bp, name);
	    if ((fp = fopen (buf, mode)) != NULL)
		return fp;
	    if (*p)
		p++;
	}
    }
    return NULL;
}/* os_path_open */
