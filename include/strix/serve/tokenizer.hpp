#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include <stdexcept>
#include <utility>

namespace strix::serve {

// Device-neutral tokenizer operations used at request and token-piece boundaries.
// Optional metadata must refuse explicitly; absence is never an invented token ID.
class ApiTokenizer {
public:
    using TokenType = BpeTokenizer::TokenType;
    virtual ~ApiTokenizer() = default;
    virtual std::vector<std::uint32_t> encode(std::string_view, bool parse_special = true) const = 0;
    virtual std::string decode(std::span<const std::uint32_t>, bool skip_control = false) const = 0;
    virtual std::string token_bytes(std::uint32_t) const = 0;
    virtual std::size_t vocab_size() const = 0;
    virtual std::uint32_t find(std::string_view) const = 0;
    virtual TokenType type(std::uint32_t) const {
        throw std::invalid_argument("tokenizer does not expose token types");
    }
    virtual std::uint32_t eos_token() const {
        throw std::invalid_argument("tokenizer does not expose an EOS token");
    }
};

// An explicit binding keeps the legacy ApiService(config, {}, ...) constructor
// unambiguous for health/deployment-only services without a tokenizer.
struct ApiTokenizerBinding {
    explicit ApiTokenizerBinding(std::shared_ptr<const ApiTokenizer> tokenizer) : value(std::move(tokenizer)) {}
    std::shared_ptr<const ApiTokenizer> value;
};

// Delegates every operation; asset loading and tokenizer algorithms stay owned
// by BpeTokenizer. The existing ApiService constructor uses this adapter.
class BpeApiTokenizer final : public ApiTokenizer {
public:
    explicit BpeApiTokenizer(std::shared_ptr<const BpeTokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {
        if (!tokenizer_) throw std::invalid_argument("API tokenizer is required");
    }
    std::vector<std::uint32_t> encode(std::string_view text, bool special = true) const override { return tokenizer_->encode(text, special); }
    std::string decode(std::span<const std::uint32_t> ids, bool skip = false) const override { return tokenizer_->decode(ids, skip); }
    std::string token_bytes(std::uint32_t id) const override { return tokenizer_->token_bytes(id); }
    std::size_t vocab_size() const override { return tokenizer_->vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return tokenizer_->find(text); }
    TokenType type(std::uint32_t id) const override { return tokenizer_->type(id); }
    std::uint32_t eos_token() const override { return tokenizer_->eos_token(); }
private:
    std::shared_ptr<const BpeTokenizer> tokenizer_;
};

} // namespace strix::serve
