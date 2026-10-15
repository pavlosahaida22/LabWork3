from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- налаштування
st.set_page_config(page_title="Titanic: прогноз виживання", page_icon="🚢", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "titanic_mlp.keras"
PREP_PATH = BASE_DIR / "models" / "titanic_preprocessor.joblib"

# Ознаки в тому ж складі, що й під час навчання (ЛР № 1)
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone", "HasCabin"]

TITLES = {"Mr": "Mr – пан", "Mrs": "Mrs – пані (заміжня)", "Miss": "Miss – панна",
          "Master": "Master – хлопчик", "Rare": "Rare – рідкісний титул (Dr, Rev, Col…)"}
PORTS = {"S": "Саутгемптон (S)", "C": "Шербур (C)", "Q": "Квінстаун (Q)"}

# Швидкі приклади для перевірки роботи моделі
PRESETS = {
    "— власні значення —": None,
    "Жінка, 1 клас, з каютою": dict(sex="female", title="Mrs", age_unknown=False, age=29, pclass=1,
                                    fare=90.0, embarked="C", has_cabin=True, sibsp=0, parch=1),
    "Чоловік, 3 клас, подорожує сам": dict(sex="male", title="Mr", age_unknown=False, age=25, pclass=3,
                                           fare=7.25, embarked="S", has_cabin=False, sibsp=0, parch=0),
    "Хлопчик, 2 клас, з родиною": dict(sex="male", title="Master", age_unknown=False, age=6, pclass=2,
                                       fare=30.0, embarked="S", has_cabin=False, sibsp=1, parch=2),
    "Чоловік, 1 клас, вік невідомий": dict(sex="male", title="Mr", age_unknown=True, age=40, pclass=1,
                                           fare=70.0, embarked="C", has_cabin=True, sibsp=0, parch=0),
}

DEFAULTS = dict(sex="female", title="Miss", age_unknown=False, age=30, pclass=3,
                fare=32.0, embarked="S", has_cabin=False, sibsp=0, parch=0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    preset = PRESETS[st.session_state["preset"]]
    if preset:
        st.session_state.update(preset)


# ------------------------------------------------- завантаження (з кешуванням)
@st.cache_resource(show_spinner="Завантаження моделі…")
def load_artifacts():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH), joblib.load(PREP_PATH)


def build_row(pclass, sex, age, sibsp, parch, fare, embarked, title, has_cabin) -> pd.DataFrame:
    """Формує рядок ознак так само, як в інженерії ознак ЛР № 1."""
    family_size = sibsp + parch + 1
    return pd.DataFrame([{
        "Pclass": pclass, "Sex": sex,
        "Age": np.nan if age is None else float(age),   # пропуск заповнить імпутер (медіана)
        "SibSp": sibsp, "Parch": parch, "Fare": float(fare),
        "Embarked": embarked, "Title": title,
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "HasCabin": int(has_cabin),
    }])[FEATURES]


# ------------------------------------------------------------------- інтерфейс
st.title("Прогноз виживання пасажира «Титаніка»")
st.caption("Лабораторна робота № 3 · Streamlit + TensorFlow/Keras · модель з ЛР № 1")

if not (MODEL_PATH.exists() and PREP_PATH.exists()):
    st.error("Не знайдено `models/titanic_mlp.keras` та/або `models/titanic_preprocessor.joblib`.")
    st.stop()

model, preprocessor = load_artifacts()

with st.sidebar:
    st.header("(i) Про модель")
    st.markdown(
        "**Тип:** багатошаровий перцептрон\n\n"
        "**Архітектура:** 22 входи → Dense(32, ReLU) → Dropout(0.3) → "
        "Dense(16, ReLU) → Dropout(0.2) → Dense(1, sigmoid)\n\n"
        f"**Параметрів:** {model.count_params():,}\n\n"
        "**Задача:** бінарна класифікація (вижив / не вижив)\n\n"
        "**Передобробка:** імпутація (медіана / мода), стандартизація числових ознак, "
        "One-Hot кодування категоріальних."
    )
    st.divider()
    st.selectbox("Швидкий приклад", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Оберіть готовий приклад – поля форми заповняться автоматично.")

st.subheader("Дані пасажира")

with st.form("titanic_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Пасажир")
        sex = st.radio("Стать", ["female", "male"], key="sex", horizontal=True,
                       format_func=lambda x: "Жінка" if x == "female" else "Чоловік")
        title = st.selectbox("Звертання", list(TITLES), key="title", format_func=TITLES.get)
        age_unknown = st.checkbox("Вік невідомий", key="age_unknown",
                                  help="Пропуск буде заповнено медіаною, як і під час навчання")
        age = st.slider("Вік", 0, 80, key="age", disabled=age_unknown)
    with c2:
        st.markdown("##### Поїздка")
        pclass = st.radio("Клас квитка", [1, 2, 3], key="pclass", horizontal=True)
        fare = st.number_input("Вартість квитка (£)", 0.0, 520.0, key="fare", step=1.0)
        embarked = st.selectbox("Порт посадки", list(PORTS), key="embarked", format_func=PORTS.get)
        has_cabin = st.checkbox("Номер каюти відомий", key="has_cabin")
    with c3:
        st.markdown("##### Родина на борту")
        sibsp = st.number_input("Брати/сестри/чоловік/дружина", 0, 8, key="sibsp")
        parch = st.number_input("Батьки/діти", 0, 6, key="parch")
        st.info(f"Розмір родини: **{sibsp + parch + 1}**  \nПодорожує сам: **{'так' if sibsp + parch == 0 else 'ні'}**")
    submitted = st.form_submit_button("🔮 Спрогнозувати", width="stretch", type="primary")

if submitted:
    row = build_row(pclass, sex, None if age_unknown else age, sibsp, parch,
                    fare, embarked, title, has_cabin)
    x = preprocessor.transform(row)
    prob = float(model.predict(x, verbose=0)[0, 0])

    st.divider()
    st.subheader("Результат")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Ймовірність виживання", f"{prob * 100:.1f} %")
        st.metric("Ймовірність загибелі", f"{(1 - prob) * 100:.1f} %")
    with r2:
        st.progress(prob, text=f"Виживання: {prob * 100:.1f} %")
        if prob >= 0.5:
            st.success("Модель прогнозує: пасажир **вижив би**.")
        else:
            st.error("Модель прогнозує: пасажир **не вижив би**.")
        st.caption("Поріг класифікації: 0.5")

    with st.expander("🔍 Вхідні ознаки та вектор після передобробки"):
        st.write("Ознаки, сформовані з введених даних:")
        st.dataframe(row, hide_index=True, width="stretch")
        st.write(f"Вектор, який отримала модель ({x.shape[1]} значень):")
        st.dataframe(pd.DataFrame(x, columns=[n.split("__")[1] for n in preprocessor.get_feature_names_out()]).round(3),
                     hide_index=True, width="stretch")
else:
    st.info("Заповніть форму та натисніть **«Спрогнозувати»**.")
