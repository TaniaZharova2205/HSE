import streamlit as st
import httpx
import asyncio
import json


BASE_URL = "http://127.0.0.1:8000/api/v1/models"

def generate_large_training_data(rows: int, features: int):
    """Создать большие данные для обучения."""
    return [
        {
            "X": [[j + i for j in range(features)] for i in range(rows)],
            "y": [sum(range(features)) + i for i in range(rows)],
            "config": {
                "hyperparameters": {"fit_intercept": True},
                "id": f"model_{rows}_{features}"
            },
        },
        {
            "X": [[j * i for j in range(features)] for i in range(rows)],
            "y": [sum(range(features)) * i for i in range(rows)],
            "config": {
                "hyperparameters": {"fit_intercept": True},
                "id": f"model_{rows}_{features}"
            },
        },
    ]

async def train_model(data):
    """Отправка данных для обучения модели."""
    async with httpx.AsyncClient() as client:
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
        training_data = st.text_area(
            "💾 Данные для обучения двух моделей:",
            value=json.dumps(
                generate_large_training_data(1000, 100000),
                indent=2,
            ),
        )


        if st.button("💃 Начать обучение двух моделей"):
            with st.spinner("✨ Обучение моделей..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(train_model(json.loads(training_data)))
                    st.success("✅ Обучение завершено!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Загрузка модели":
        st.header("📤 Загрузка моделей")
        load_data = st.text_area(
            "Введите данные для загрузки моделей:",
            value=json.dumps(
                [{"id": "model_1"}, {"id": "model_2"}],
                indent=2,
            ),
        )

        if st.button("📤 Загрузить модели"):
            with st.spinner("📤 Загрузка моделей..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(load_model(json.loads(load_data)))
                    st.success("✅ Модели загружены!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")

    elif menu == "Предсказание":
        st.header("🔮 Предсказание")
        prediction_data = st.text_area(
            "📊 Данные для предсказания:",
            value=json.dumps(
                [
                    {"id": "model_1", "X": [[10, 11, 12, 13], [20, 21, 22, 23]]},
                    {"id": "model_2", "X": [[30, 32, 34, 36], [40, 42, 44, 46]]},
                ],
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
