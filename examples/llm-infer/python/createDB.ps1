$body = @{
    folder_path = "E:\RagDocs"
    output_dir = "E:\RAG\output"
    max_chunk_size = 1000
    similarity_threshold = 0.7
    use_gpu = $true
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/build" -Method Post -Body $body -ContentType "application/json"
$jobId = $response.job_id

Write-Host "Job started with ID: $jobId"

# Monitor job status
do {
    Start-Sleep -Seconds 5
    $status = Invoke-RestMethod -Uri "http://localhost:8000/status/$jobId" -Method Get
    Write-Host "Status: $($status.status), Progress: $([math]::Round($status.progress * 100))%, Message: $($status.message)"
} until ($status.completed -eq $true)

if ($status.status -eq "completed") {
    Write-Host "Success! Created $($status.chunk_count) chunks from $($status.document_count) documents."
} else {
    Write-Host "Failed: $($status.message)"
}