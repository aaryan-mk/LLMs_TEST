<template>
  <div>
    <h2>General AI Assistant</h2>
    <!-- General Chat Display -->
    <div class="chat-window">
      <div
        v-for="(chat, index) in chatHistory"
        :key="index"
        class="chat-message"
        v-html="formatMessage(chat.content)"
      >
      </div>
    </div>

    <!-- General Chat Controls -->
    <div class="controls">
      <input v-model="query" placeholder="Ask a question..." />
      <button @click="fetchResponse">Ask</button>
    </div>

    <h2>Course Recommendations</h2>
    <!-- Course Recommendations Chat Display -->
    <div class="chat-window">
      <div
        v-for="(chat, index) in courseChatHistory"
        :key="index"
        class="chat-message"
        v-html="formatMessage(chat.content)"
      >
      </div>
    </div>

    <!-- Course Recommendations Controls -->
    <div class="controls">
      <input v-model="courseQuery" placeholder="Enter your course preferences..." />
      <button @click="fetchCourseRecommendation">Get Recommendations</button>
    </div>

    <!-- Week Notes Buttons -->
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
      query: "",                    // User query for general AI assistant
      courseQuery: "",              // User query for course recommendation
      chatHistory: [],              // History of the general chat (user + bot)
      courseChatHistory: [],        // History of the course recommendation chat (user + bot)
      loading: false
    };
  },
  methods: {
    // Fetch general AI assistant response, including the chat history
    async fetchResponse() {
      if (!this.query) return;
      this.loading = true;
      try {
        // Combine query with history for the request
        const combinedInput = this.chatHistory.map((item) => item.content).join(" ") + " " + this.query;

        const res = await fetch("http://127.0.0.1:5000/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: combinedInput,  // Send combined input
            history: this.chatHistory // Sending the chat history with the query
          })
        });

        const data = await res.json();

        // Add user query and bot response to general chat history
        this.chatHistory.push({ role: "User", content: this.query });
        this.chatHistory.push({ role: "Bot", content: data.response });

        // Clear the query input field after sending the request
        this.query = "";
      } catch (error) {
        console.error("Error fetching response:", error);
      }
      this.loading = false;
    },

    // Fetch course recommendations, including course chat history
    async fetchCourseRecommendation() {
      if (!this.courseQuery) return;
      this.loading = true;
      try {
        // Combine query with history for the request
        const combinedInput = this.courseChatHistory.map((item) => item.content).join(" ") + " " + this.courseQuery;

        const res = await fetch("http://127.0.0.1:5000/recommend-courses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: combinedInput,  // Send combined input
            history: this.courseChatHistory // Send course chat history along with query
          })
        });

        const data = await res.json();

        // Add user course query and bot recommendation to course chat history
        this.courseChatHistory.push({ role: "User", content: this.courseQuery });
        this.courseChatHistory.push({ role: "Bot", content: data.response });

        // Clear the course query input field
        this.courseQuery = "";
      } catch (error) {
        console.error("Error fetching course recommendations:", error);
      }
      this.loading = false;
    },

    // Generate week notes based on selected week
    async generateWeekNotes(week) {
      this.loading = true;
      try {
        const res = await fetch(`http://127.0.0.1:5000/generate-notes/${week}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });
        const data = await res.json();

        // Add generated notes as bot messages to general chat history
        this.chatHistory.push({ role: "Bot", content: data.response });
      } catch (error) {
        console.error(`Error generating ${week} notes:`, error);
      }
      this.loading = false;
    },

    // Method to format message content with Markdown to HTML
    formatMessage(content) {
      return content
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // Bold text (**) to <strong>
        .replace(/\*(.*?)\*/g, "<em>$1</em>") // Italic text (*) to <em>
        .replace(/\n/g, "<br/>"); // Newlines to <br/> for line breaks
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
