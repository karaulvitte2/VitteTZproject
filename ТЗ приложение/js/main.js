// static/js/main.js

document.addEventListener("DOMContentLoaded", () => {
    // Логика для страницы history.html
    const historyForm = document.querySelector(".history-form");
    if (!historyForm) return;

    const table = historyForm.querySelector(".table-history");
    const checkboxes = Array.from(
        historyForm.querySelectorAll('input[type="checkbox"][name="log_ids"]')
    );

    // Если добавишь чекбокс "выбрать всё" в заголовок (id="select_all_logs")
    const selectAll = historyForm.querySelector("#select_all_logs");

    // Элемент для отображения количества выбранных (если захочешь)
    let counter = document.querySelector(".selected-count");
    if (!counter) {
        counter = document.createElement("div");
        counter.className = "selected-count";
        counter.style.marginTop = "0.4rem";
        counter.style.fontSize = "0.8rem";
        counter.style.color = "rgba(209, 213, 219, 0.9)";
        historyForm.appendChild(counter);
    }

    const updateCounter = () => {
        const selected = checkboxes.filter(cb => cb.checked).length;
        if (selected === 0) {
            counter.textContent = "Ни одной записи не выбрано.";
        } else if (selected === 1) {
            counter.textContent = "Выбрана 1 запись для включения в ТЗ.";
        } else {
            counter.textContent = `Выбрано ${selected} записей для включения в ТЗ.`;
        }
    };

    updateCounter();

    checkboxes.forEach(cb => {
        cb.addEventListener("change", updateCounter);
    });

    if (selectAll) {
        selectAll.addEventListener("change", () => {
            const value = selectAll.checked;
            checkboxes.forEach(cb => {
                cb.checked = value;
            });
            updateCounter();
        });
    }
});
