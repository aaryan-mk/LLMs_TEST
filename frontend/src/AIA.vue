<template>
    <div>
      <h2>AI Assistant</h2>
      
      <!-- Chat Display -->
      <div class="chat-window">
        <div v-for="(chat, index) in chatHistory" :key="index" class="chat-message">
          <strong>{{ chat.role }}:</strong> {{ chat.content }}
        </div>
      </div>
  
      <!-- User Input -->
      <input v-model="query" placeholder="Ask a question...">
      <button @click="fetchResponse">Ask</button>
      <div v-if="loading">Loading...</div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        query: "",
        chatHistory: [],
        loading: false
      };
    },
    methods: {
      async fetchResponse() {
        if (!this.query) return;
        this.loading = true;
        try {
          const res = await fetch("http://127.0.0.1:5000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: this.query })
          });
          const data = await res.json();
          
          // Append new message to chat history
          this.chatHistory.push({ role: "User", content: this.query });
          this.chatHistory.push({ role: "Bot", content: data.response });
          this.query = "";  // Clear input field
        } catch (error) {
          console.error("Error fetching response:", error);
        }
        this.loading = false;
      }
    }
  };
  </script>
  
  <style>
  .chat-window {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 10px;
    margin-bottom: 10px;
  }
  .chat-message {
    margin-bottom: 5px;
  }
  </style>  