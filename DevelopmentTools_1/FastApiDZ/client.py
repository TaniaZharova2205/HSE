import streamlit as st
import httpx
import asyncio
import json
from sklearn.datasets import make_blobs

BASE_URL = "http://127.0.0.1:8000"

row = 1000000
feature = 40
center = 2

def generate_data(n_samples, n_features, centers):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
    return X,y

def generate_large_training_data(rows: int, features: int, centers: int):
    """Создать большие данные для обучения."""
    model1 = generate_data(rows, features, centers)
    model2 = generate_data(rows, features, centers)
    return [
        {
            "X": model1[0].tolist(),
            "y": model1[1].tolist(),
            "config": {
                "hyperparameters": {"fit_intercept": True},
                "id": f"model1"
            },
        },
        {
            "X": model2[0].tolist(),
            "y": model2[1].tolist(),
            "config": {
                "hyperparameters": {"fit_intercept": True},
                "id": f"model2"
            },
        },
    ]

def generate_prediction_data(features: int):
    """Создать данные для предсказания."""
    return {"id": "model1", "X": [[i for i in range(features)]]}


async def train_model(data):
    """Отправка данных для обучения модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        response = await client.post(f"{BASE_URL}/fit", json=data)
        response.raise_for_status()
        return response.json()


async def load_model(data):
    """Загрузка модели."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/load", json=data)
        response.raise_for_status()
        return response.json()


async def predict_model(data):
    """Асинхронное предсказание."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()


async def list_models():
    """Получение списка моделей."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/list_models")
        response.raise_for_status()
        return response.json()


async def remove_all_models():
    """Удаление всех моделей."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{BASE_URL}/remove_all")
        response.raise_for_status()
        return response.json()


def render_header(title: str, subtitle: str):
    st.title(f"💅🏻 {title}")
    st.caption(subtitle)


def render_menu():
    st.sidebar.title("🎨 Навигация")
    return st.sidebar.radio(
        "👑 Выберите действие:",
        ["Обучение", "Загрузка модели", "Предсказание", "Список моделей", "Удаление моделей"],
    )


def app():
    render_header("Model Management Interface", "Стильный интерфейс для управления ML-моделями")
    menu = render_menu()

    if menu == "Обучение":
        st.header("🔥 Обучение двух моделей")

        if st.button("💃 Начать обучение двух моделей"):
            with st.spinner("✨ Обучение моделей..."):
                try:
                    training_data = generate_large_training_data(row, feature, center)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(train_model(training_data))
                    st.success("✅ Обучение завершено!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Загрузка модели":
        st.header("📤 Загрузка модели")
        load_data = st.text_area(
            "Введите данные для загрузки модели:",
            value=json.dumps(
                {"id": "model1"},
                indent=2,
            ),
        )

        if st.button("📤 Загрузить модели"):
            with st.spinner("📤 Загрузка модели..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(load_model(json.loads(load_data)))
                    st.success("✅ Модель загружены!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Предсказание":
        st.header("🔮 Предсказание")
        prediction_data = st.text_area(
            "📊 Данные для предсказания:",
            value=json.dumps(
                generate_prediction_data(feature),
                indent=2,
            ),
        )

        if st.button("✨ Сделать предсказание"):
            with st.spinner("🔮 Выполнение предсказаний..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(predict_model(json.loads(prediction_data)))
                    st.success("✅ Предсказания получены!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Список моделей":
        st.header("📂 Список моделей")
        if st.button("📋 Получить список моделей"):
            with st.spinner("📂 Получение списка моделей..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    models = loop.run_until_complete(list_models())
                    st.success("✅ Список моделей получен!")
                    st.json(models)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Удаление моделей":
        st.header("🗑️ Удаление всех моделей")
        if st.button("❌ Удалить все модели"):
            with st.spinner("🗑️ Удаление моделей..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(remove_all_models())
                    st.success("✅ Все модели удалены!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")


if __name__ == "__main__":
    app()