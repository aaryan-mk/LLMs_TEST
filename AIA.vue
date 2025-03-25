<template>
  <div>
    <h2>AI Assistant</h2>

    <!-- Chat Display -->
    <div class="chat-window">
      <div v-for="(chat, index) in chatHistory" :key="index" class="chat-message">
        <strong>{{ chat.role }}:</strong> {{ chat.content }}
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <input v-model="query" placeholder="Ask a question..." />
      <button @click="fetchResponse">Ask</button>
    </div>
    
    <!-- Week Buttons -->
    <div class="controls">
      <button @click="generateWeekNotes('week1')">Generate Week1 Notes</button>
      <button @click="generateWeekNotes('week2')">Generate Week2 Notes</button>
      <button @click="generateWeekNotes('week3')">Generate Week3 Notes</button>
      <button @click="generateWeekNotes('week4')">Generate Week4 Notes</button>
    </div>

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
        this.chatHistory.push({ role: "User", content: this.query });
        this.chatHistory.push({ role: "Bot", content: data.response });
        this.query = "";
      } catch (error) {
        console.error("Error fetching response:", error);
      }
      this.loading = false;
    },

    async generateWeekNotes(week) {
      this.loading = true;
      try {
        const res = await fetch(`http://127.0.0.1:5000/generate-notes/${week}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });
        const data = await res.json();
        this.chatHistory.push({ role: "Bot", content: data.response });
      } catch (error) {
        console.error(`Error generating ${week} notes:`, error);
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
.controls {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}
</style>
