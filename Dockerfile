# syntax=docker/dockerfile:1
FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Runtime: debian-slim instead of distroless so ffmpeg is available for
# opus response_format transcoding (MiniMax upstream cannot emit opus).
FROM debian:12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/voice-gateway /usr/local/bin/voice-gateway
ENV LISTEN_ADDR=0.0.0.0:8090
EXPOSE 8090
RUN useradd --system --user-group --no-create-home --home-dir /nonexistent nonroot
USER nonroot
ENTRYPOINT ["/usr/local/bin/voice-gateway"]
