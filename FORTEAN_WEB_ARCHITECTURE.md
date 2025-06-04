# Fortean Chat Web Architecture

## Overview

A secure, efficient architecture for serving your Fortean AI chat to collaborators:

```
[Browser] <--HTTPS--> [ngrok] <---> [Mac Mini]
                                      |
                                      ├── Rust Web Server (port 8080)
                                      │   └── Serves WASM app + static files
                                      │
                                      └── Ollama API (port 11434)
                                          └── Fortean model inference
```

## Architecture Components

### 1. Frontend: Rust WebAssembly App
- **Framework**: Yew or Leptos (reactive web frameworks)
- **Features**:
  - Password-protected entry
  - Real-time chat interface
  - Streaming responses
  - Session management with JWT
- **Security**: All auth happens client-side first, then validated server-side

### 2. Backend: Rust Web Server
- **Framework**: Axum or Actix-web
- **Responsibilities**:
  - Serve WASM bundle and static assets
  - Proxy authenticated requests to Ollama
  - Rate limiting per session
  - CORS handling for ngrok
- **Why Rust all the way**: Single language, excellent performance, built-in safety

### 3. Inference: Ollama
- Already running on Mac Mini
- Exposed only to localhost
- Rust backend proxies requests with auth

### 4. Tunnel: ngrok
- HTTPS encryption
- Built-in request inspection
- Rate limiting at edge
- Custom domain (optional)

## Security Model

### Password Protection
1. User enters password in WASM app
2. Password hashed client-side (Argon2)
3. Hash sent with each API request
4. Server validates hash against stored hash
5. JWT issued for session management

### DOS Protection
1. **ngrok level**: Rate limiting, IP blocking
2. **Rust server level**: 
   - Per-session rate limits
   - Request queuing
   - Timeout handling
3. **Ollama level**: Single request processing

## Implementation Plan

### Phase 1: Rust API Server
```rust
// Cargo.toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "fs"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.11", features = ["stream"] }
argon2 = "0.5"
jsonwebtoken = "9"
```

Key endpoints:
- `POST /auth` - Validate password, issue JWT
- `POST /chat` - Stream chat responses (requires valid JWT)
- `GET /` - Serve WASM app

### Phase 2: WASM Chat App
```rust
// Frontend Cargo.toml
[dependencies]
yew = "0.21"
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = "0.3"
gloo-net = "0.5"
gloo-storage = "0.3"
```

Components:
- Password entry screen
- Chat interface with streaming
- Message history
- Error handling

### Phase 3: Deployment Setup
```bash
# Start Ollama
ollama serve

# Start Rust server
cargo run --release

# Start ngrok with auth
ngrok http 8080 --auth="username:password"
```

## Project Structure
```
fortean-chat-web/
├── Cargo.toml          # Workspace
├── server/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs     # Axum server
│       ├── auth.rs     # Password validation
│       └── ollama.rs   # Ollama client
├── client/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs     # Yew app entry
│       ├── components/
│       │   ├── auth.rs
│       │   └── chat.rs
│       └── api.rs      # Server communication
└── dist/               # Built WASM + assets
```

## Advantages of This Architecture

1. **Single Language**: Rust everywhere (except Ollama)
2. **Performance**: WASM is fast, Rust server is efficient
3. **Security**: Memory safe, no injection vulnerabilities
4. **Simplicity**: One server process, built-in static serving
5. **Scalability**: Can handle many concurrent users with async Rust

## Alternative: Simpler Python + Rust Hybrid

If you prefer to keep Python for the API:
1. Python FastAPI server for Ollama interaction
2. Rust server just serves WASM and proxies to Python
3. Slightly more complex but allows reusing Python chat code

Which approach would you prefer? I can provide detailed implementation code for either path.