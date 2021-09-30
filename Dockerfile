FROM python:3.9-slim-bullseye as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.1.7

RUN python -m pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"

RUN python -m venv /venv
COPY pyproject.toml poetry.lock ./

RUN . /venv/bin/activate && poetry install --no-root $(test "$YOUR_ENV" == production && echo "--no-dev")

COPY . .
RUN . /venv/bin/activate && poetry build

FROM base as final

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .
ENV PATH="/venv/bin:$PATH"
RUN . /venv/bin/activate && pip install *.whl
CMD ["python", "-m", "gdrf"]
