document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-upload");
    const uploadStatus = document.getElementById("upload-status");
    const analyzeButton = document.getElementById("analyze-btn");
    const analyzeStatus = document.getElementById("analyze-status");
    const resultContainer = document.getElementById("result-container");

    let uploadedFile = null;

    // Handle file upload
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = "Please select a file to upload.";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        uploadStatus.textContent = "Uploading...";
        try {
            const response = await fetch("/upload/", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                uploadStatus.textContent = data.message;
                uploadedFile = file;
                analyzeButton.disabled = false;
            } else {
                uploadStatus.textContent = `Error: ${data.detail || "Unknown error"}`;
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
        }
    });

    // Handle document analysis
    analyzeButton.addEventListener("click", async () => {
        if (!uploadedFile) {
            analyzeStatus.textContent = "Please upload a file first.";
            return;
        }

        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("query", "Comprehensive document analysis");

        analyzeStatus.textContent = "Analyzing document...";
        try {
            const response = await fetch("/analyze/", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                analyzeStatus.textContent = data.message;
                
                // Display search results and the download link for the PDF report
                resultContainer.innerHTML = `
                    <h3>Analysis Results</h3>
                    <p>Report generated: <a href="/download-report/${data.report_path}" target="_blank">Download Report (PDF)</a></p>
                    <h4>Search Results:</h4>
                    <pre>${JSON.stringify(data.search_results, null, 2)}</pre>
                `;
            } else {
                analyzeStatus.textContent = `Error: ${data.detail || "Unknown error"}`;
            }
        } catch (error) {
            analyzeStatus.textContent = `Error: ${error.message}`;
        }
    });
});
