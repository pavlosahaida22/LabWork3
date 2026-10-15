# ЛР № 3 – Веб-застосунок Streamlit для моделі Titanic

Застосунок використовує модель з ЛР № 1: MLP 22→32→16→1 (sigmoid) для прогнозу виживання пасажира «Титаніка».

## Структура проєкту
```
lab3_streamlit_app/
├── app.py                          # застосунок Streamlit
├── requirements.txt                # залежності (для Streamlit Cloud)
├── README.md
├── .gitignore
└── models/
    ├── titanic_mlp.keras           # модель з ЛР № 1
    └── titanic_preprocessor.joblib # конвеєр передобробки з ЛР № 1
```

## Локальний запуск
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit hello                    # перевірка встановлення
streamlit run app.py
```
Відкриється http://localhost:8501.

> `scikit-learn==1.6.1` зафіксовано, бо конвеєр `titanic_preprocessor.joblib` збережено з цією версією
> (в інших версіях він може не завантажитись).

## Розгортання на Streamlit Community Cloud
1. Створіть репозиторій на GitHub та завантажте вміст папки проєкту (разом із `models/`):
   ```bash
   git init
   git add .
   git commit -m "Lab 3: Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<ваш-логін>/<репозиторій>.git
   git push -u origin main
   ```
2. Відкрийте https://share.streamlit.io → увійдіть через GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Вкажіть репозиторій, гілку `main`, Main file path: `app.py`.
5. (Advanced settings → Python 3.11 або 3.12.)
6. **Deploy**, дочекайтеся встановлення залежностей і скопіюйте посилання `https://<назва>.streamlit.app`.

## Тестові приклади (п. 3.2.10)
У бічній панелі є список «Швидкий приклад». Орієнтовні результати:

| Приклад | Ймовірність виживання |
|---|---|
| Жінка, 1 клас, з каютою, 29 років | ≈ 95 % |
| Чоловік, 3 клас, сам, 25 років | ≈ 10 % |
| Хлопчик, 2 клас, з родиною, 6 років | ≈ 53 % |
| Чоловік, 1 клас, вік невідомий | ≈ 55 % |
