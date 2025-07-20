/**
 * AccesLibre - Script JavaScript principal
 * Certification RNCP 38616 - Grégory LE TERTE
 */

document.addEventListener('DOMContentLoaded', function() {
    // Référence aux éléments du DOM
    const fileInput = document.getElementById('imageUpload');
    const form = fileInput ? fileInput.closest('form') : null;
    const submitButton = form ? form.querySelector('button[type="submit"]') : null;
    
    // Gestion de l'upload de fichier
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            // Afficher le nom du fichier sélectionné
            const fileName = e.target.files[0] ? e.target.files[0].name : '';
            const fileInfo = document.createElement('div');
            fileInfo.className = 'mt-2 text-muted small';
            fileInfo.innerHTML = fileName ? `<i class="fas fa-file-image me-1"></i> ${fileName}` : '';
            
            // Supprimer l'ancienne info si elle existe
            const oldInfo = fileInput.parentElement.parentElement.nextElementSibling;
            if (oldInfo && oldInfo.classList.contains('text-muted')) {
                oldInfo.remove();
            }
            
            // Ajouter la nouvelle info
            if (fileName) {
                fileInput.parentElement.parentElement.after(fileInfo);
            }
        });
    }
    
    // Gestion du formulaire
    if (form) {
        form.addEventListener('submit', function(e) {
            // Vérifier si un fichier a été sélectionné
            if (fileInput.files.length === 0) {
                e.preventDefault();
                alert('Veuillez sélectionner une image à analyser.');
                return false;
            }
            
            // Afficher l'état de chargement
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyse en cours...';
            }
            
            return true;
        });
    }
    
    // Animation pour les barres de progression
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = width;
        }, 200);
    });
});
