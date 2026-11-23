#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

namespace strix::serve {

extern const BpeTokenizer::HfIdentity inkling2_mini_tokenizer_identity;

// Text binding of the Inkling 2 Mini checkpoint's chat_template.jinja. Effort is an
// alias, a number in [0, 0.99], or null for the default. Messages support text parts
// and the template's tool protocol; image parts render only through the media entry points.
// The checkpoint template emits no BOS or assistant generation prefix.
[[nodiscard]] std::string render_inkling2_chat(const Json& messages, const Json& tools = {},
                                              const Json& reasoning_effort = {});
// The same rendering with image parts (image, input_image, image_url; after media::normalize {"type": "image"}) as
// the template's <|content_image|><|unused_200054|> frame (user and tool messages only), for the caller to expand
// into image rows. render_inkling2_chat keeps refusing image parts.
[[nodiscard]] std::string render_inkling2_chat_media(const Json& messages, const Json& tools = {},
                                                    const Json& reasoning_effort = {});

[[nodiscard]] std::string render_inkling2_chat_request(std::string_view json);
// The same request with image parts (after media::normalize: {"type": "image", ...}) rendered as image frames.
[[nodiscard]] std::string render_inkling2_chat_media_request(std::string_view json);
// The template text of the image placeholder token (one id).
inline constexpr std::string_view kInkling2ImagePlaceholder = "<|unused_200054|>";

}  // namespace strix::serve
