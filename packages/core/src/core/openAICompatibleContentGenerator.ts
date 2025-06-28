/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
  FinishReason,
  ContentListUnion,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export interface OpenAIConfig {
  endpoint: string;
  model: string;
  apiKey?: string;
}

// Create a custom response class that matches the expected interface
class OpenAIGenerateContentResponse {
  candidates: any[];
  usageMetadata?: any;

  constructor(candidates: any[], usageMetadata?: any) {
    this.candidates = candidates;
    this.usageMetadata = usageMetadata;
  }

  get text(): string {
    const parts = this.candidates?.[0]?.content?.parts;
    if (!parts) {
      return '';
    }
    return parts
      .map((part: any) => part.text)
      .filter((text: any) => typeof text === 'string')
      .join('');
  }

  get data(): string {
    return '';
  }

  get functionCalls(): any[] {
    return [];
  }

  get executableCode(): string {
    return '';
  }

  get codeExecutionResult(): string {
    return '';
  }
}

export class OpenAICompatibleContentGenerator implements ContentGenerator {
  constructor(private config: OpenAIConfig) {}

  private convertToOpenAIMessages(contents: ContentListUnion): any[] {
    // Handle string input
    if (typeof contents === 'string') {
      return [{ role: 'user', content: contents }];
    }
    
    // Handle single Content object
    if (!Array.isArray(contents)) {
      const content = contents as Content;
      const role = content.role === 'model' ? 'assistant' : content.role;
      const parts = content.parts || [];
      
      const textParts = parts
        .filter((part: Part) => 'text' in part)
        .map((part: Part) => (part as any).text)
        .join('\n');
      
      return [{
        role,
        content: textParts,
      }];
    }
    
    // Handle array of Content objects
    return contents.map((content: any) => {
      const role = content.role === 'model' ? 'assistant' : content.role;
      const parts = content.parts || [];
      
      // Combine all text parts into a single message
      const textParts = parts
        .filter((part: Part) => 'text' in part)
        .map((part: Part) => (part as any).text)
        .join('\n');
      
      return {
        role,
        content: textParts,
      };
    });
  }

  private convertFromOpenAIResponse(openAIResponse: any): GenerateContentResponse {
    const choice = openAIResponse.choices?.[0];
    const messageContent = choice?.message?.content || '';
    
    if (!choice) {
      return new OpenAIGenerateContentResponse([]) as any;
    }

    const content: Content = {
      role: 'model',
      parts: [{ text: messageContent }],
    };

    const candidates = [{
      content,
      index: 0,
      finishReason: choice.finish_reason === 'stop' ? FinishReason.STOP : FinishReason.OTHER,
    }];

    const usageMetadata = openAIResponse.usage ? {
      promptTokenCount: openAIResponse.usage.prompt_tokens,
      candidatesTokenCount: openAIResponse.usage.completion_tokens,
      totalTokenCount: openAIResponse.usage.total_tokens,
    } : undefined;

    return new OpenAIGenerateContentResponse(candidates, usageMetadata) as any;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertToOpenAIMessages(request.contents);
    
    // Add system instruction if provided
    if (request.config?.systemInstruction) {
      messages.unshift({
        role: 'system',
        content: typeof request.config.systemInstruction === 'string' 
          ? request.config.systemInstruction 
          : (request.config.systemInstruction as any).text || '',
      });
    }

    const openAIRequest = {
      model: this.config.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      stream: false,
    };

    try {
      const response = await fetch(`${this.config.endpoint}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(openAIRequest),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const openAIResponse = await response.json();
      return this.convertFromOpenAIResponse(openAIResponse);
    } catch (error) {
      console.error('Error calling OpenAI compatible API:', error);
      throw error;
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const generator = this._generateContentStream(request);
    return generator;
  }

  private async *_generateContentStream(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const messages = this.convertToOpenAIMessages(request.contents);
    
    // Add system instruction if provided
    if (request.config?.systemInstruction) {
      messages.unshift({
        role: 'system',
        content: typeof request.config.systemInstruction === 'string' 
          ? request.config.systemInstruction 
          : (request.config.systemInstruction as any).text || '',
      });
    }

    const openAIRequest = {
      model: this.config.model,
      messages,
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      stream: true,
    };

    try {
      const response = await fetch(`${this.config.endpoint}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(openAIRequest),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last line if it's incomplete
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            
            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta;
              
              if (delta?.content) {
                const finishReason = chunk.choices?.[0]?.finish_reason;
                const candidates = [{
                  content: {
                    role: 'model',
                    parts: [{ text: delta.content }],
                  },
                  index: 0,
                  finishReason: finishReason === 'stop' ? FinishReason.STOP : undefined,
                }];
                
                yield new OpenAIGenerateContentResponse(candidates) as any;
              }
            } catch (e) {
              // Ignore parse errors for individual chunks
            }
          }
        }
      }
    } catch (error) {
      console.error('Error calling OpenAI compatible streaming API:', error);
      throw error;
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Simple approximation - you might want to use a proper tokenizer
    let text = '';
    if (typeof request.contents === 'string') {
      text = request.contents;
    } else if (Array.isArray(request.contents)) {
      text = request.contents
        .flatMap((content: Content) => content.parts || [])
        .filter((part: Part) => 'text' in part)
        .map((part: Part) => (part as any).text)
        .join(' ');
    } else {
      // Single Content object
      const content = request.contents as Content;
      text = content.parts
        ?.filter((part: Part) => 'text' in part)
        .map((part: Part) => (part as any).text)
        .join(' ') || '';
    }
    
    // Rough approximation: 1 token ≈ 4 characters
    const tokenCount = Math.ceil(text.length / 4);
    
    return {
      totalTokens: tokenCount,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // For now, return a dummy embedding
    // You would implement actual embedding API call here if needed
    return {
      embeddings: [{
        values: new Array(768).fill(0),
      }],
    };
  }
}