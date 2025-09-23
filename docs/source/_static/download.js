// Enhance download links so clicking initiates a file download instead of
// letting the browser render the notebook JSON inline.
document.addEventListener('DOMContentLoaded', () => {
  const notebookManifest = window.GRADNET_NOTEBOOKS || {};
  const links = document.querySelectorAll('a.reference.download.internal');
  links.forEach((link) => {
    const href = link.getAttribute('href');
    if (!href) {
      return;
    }

    // Ensure the anchor has a filename in its download attribute for browsers
    // that honor it without additional handling.
    const inferredName = link.getAttribute('download') || href.split('/').pop() || 'download';
    link.setAttribute('download', inferredName);

    link.addEventListener('click', (event) => {
      // Allow modifier clicks (new tab, etc.) and non-primary buttons to behave normally.
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }

      event.preventDefault();
      const downloadUrl = link.href;

      const encoded = notebookManifest[inferredName];
      if (encoded) {
        const byteString = atob(encoded);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i += 1) {
          bytes[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'application/x-ipynb+json' });
        const blobUrl = URL.createObjectURL(blob);
        const tmpLink = document.createElement('a');
        tmpLink.href = blobUrl;
        tmpLink.download = inferredName;
        tmpLink.style.display = 'none';
        document.body.appendChild(tmpLink);
        tmpLink.click();
        document.body.removeChild(tmpLink);
        URL.revokeObjectURL(blobUrl);
        return;
      }

      fetch(downloadUrl)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch ${downloadUrl}: ${response.status}`);
          }
          return response.blob();
        })
        .then((blob) => {
          const blobUrl = URL.createObjectURL(blob);
          const tmpLink = document.createElement('a');
          tmpLink.href = blobUrl;
          tmpLink.download = inferredName;
          tmpLink.style.display = 'none';
          document.body.appendChild(tmpLink);
          tmpLink.click();
          document.body.removeChild(tmpLink);
          URL.revokeObjectURL(blobUrl);
        })
        .catch(() => {
          // Fallback to default navigation if something goes wrong.
          window.location.href = downloadUrl;
        });
    });
  });
});
