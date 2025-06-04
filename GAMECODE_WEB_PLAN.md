# GameCode Web - Multi-Engine AI Chat Platform

## Vision

A flexible Rust-based web chat application that can connect to multiple AI inference engines:
- **Phase 1**: Local Ollama (Fortean model)
- **Phase 2**: AWS Bedrock
- **Phase 3**: Other providers (OpenAI, Anthropic, etc.)
- **Future**: MCP tool integration

## Architecture Overview

```
[Browser WASM App] <--HTTPS--> [ngrok] <---> [Rust Server on Mac Mini]
                                                |
                                                ├── Inference Router
                                                │   ├── Ollama Provider
                                                │   ├── Bedrock Provider (future)
                                                │   └── MCP Tools (future)
                                                │
                                                └── Static File Server
```

## Key Design Decisions

### 1. Provider Abstraction
```rust
// Common trait for all inference providers
trait InferenceProvider {
    async fn chat(&self, prompt: String, config: ChatConfig) -> Result<ChatStream>;
    fn name(&self) -> &str;
    fn available(&self) -> bool;
}
```

This allows adding new providers without changing the core architecture.

### 2. Configuration-Driven
- TOML/JSON config specifies available providers
- Runtime provider selection
- Easy to add/remove engines

### 3. Security Model
- Single shared password for simplicity
- JWT tokens for session management
- Provider-specific auth (AWS creds for Bedrock, etc.)

### 4. Repository Structure
```
../gamecode-web/
├── Cargo.toml              # Workspace
├── server/                 # Rust backend
│   ├── src/
│   │   ├── main.rs        # Axum server
│   │   ├── auth.rs        # Password & JWT
│   │   ├── providers/     # Inference engines
│   │   │   ├── mod.rs
│   │   │   ├── ollama.rs
│   │   │   └── bedrock.rs (future)
│   │   └── api.rs         # REST endpoints
│   └── Cargo.toml
├── client/                 # WASM frontend
│   ├── src/
│   │   ├── main.rs        # Yew/Leptos app
│   │   ├── components/
│   │   └── api.rs
│   └── Cargo.toml
├── config/
│   └── default.toml       # Provider configuration
└── dist/                  # Built assets
```

## Implementation Phases

### Phase 1: Ollama Integration (Current Focus)
- Basic Rust server with Axum
- Ollama provider implementation
- WASM chat interface
- Password protection
- ngrok deployment

### Phase 2: Multi-Provider Support
- Provider trait abstraction
- Configuration system
- Provider selection in UI
- Error handling per provider

### Phase 3: AWS Bedrock
- AWS SDK integration
- Credential management
- Bedrock-specific features

### Phase 4: Advanced Features
- MCP tool integration
- Conversation memory
- Multiple model selection per provider
- Usage tracking

## API Design

### Endpoints
```
POST /auth
  -> { "password": "..." }
  <- { "token": "jwt..." }

GET /providers
  <- { "providers": ["ollama", "bedrock"] }

POST /chat
  -> { "provider": "ollama", "model": "fortean", "prompt": "..." }
  <- Stream of { "text": "..." }

GET /health
  <- { "status": "ok", "providers": {...} }
```

## Development Approach

1. **Start Simple**: Get Ollama working end-to-end
2. **Abstract Early**: Design provider trait from the start
3. **Test Locally**: Full stack on Mac Mini
4. **Deploy Incrementally**: Add providers one at a time

## Questions Before We Start

1. **Frontend Framework**: Yew (more mature) or Leptos (more modern)?
2. **Streaming**: Server-Sent Events or WebSockets?
3. **Config Format**: TOML, JSON, or environment variables?
4. **Ollama Access**: Direct HTTP or use a Rust client library?

Once you confirm these choices, we can start building the foundation in `../gamecode-web`.