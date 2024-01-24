// ==========================================
// SHOW UPLOADED IMAGE
// ==========================================
/**
 * Reads the URL of the selected image file and displays it in the imageResult element.
 * @param {HTMLInputElement} input - The input element that contains the selected image file.
 */
function readURL(input) {
  const imageResult = document.getElementById("imageResult");

  if (input.files && input.files[0]) {
    const reader = new FileReader();

    reader.onload = function (e) {
      imageResult.src = e.target.result;
    };
    reader.readAsDataURL(input.files[0]);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const uploadInput = document.getElementById("upload");

  uploadInput.addEventListener("change", function () {
    readURL(uploadInput);
  });
});

// ==========================================
// SHOW UPLOADED IMAGE NAME
// ==========================================
const uploadInput = document.getElementById("upload");
const infoArea = document.getElementById("upload-label");

uploadInput.addEventListener("change", showFileName);

/**
 * Displays the name of the selected file.
 *
 * @param {Event} event - The event object triggered by the file input.
 */
function showFileName(event) {
  const fileName = event.target.files[0].name;
  infoArea.textContent = `File name: ${fileName}`;
}
