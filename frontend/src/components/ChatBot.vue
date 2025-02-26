<template>
  <section id="chatbot" class="chatbot-container">
    <h1>与车车对话</h1>
    <p>有任何问题？想了解更多车辆信息？我随时为您服务！</p>
    <div class="chat-container">
      <div class="chat-messages" ref="chatMessages">
        <div v-for="(message, index) in messages" :key="index" 
             :class="['message', message.role]">
          <div class="message-content">{{ message.content }}</div>
        </div>
        <div v-if="isTyping" class="message assistant">
          <div class="message-content typing">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
      <div class="chat-input">
        <input 
          type="text" 
          v-model="userInput" 
          @keyup.enter="sendMessage"
          :disabled="isProcessing"
          placeholder="请输入您的问题..."
        >
        <button 
          @click="sendMessage" 
          :disabled="isProcessing || !userInput.trim()"
        >
          发送
        </button>
      </div>
    </div>
  </section>
</template>

<script>
import axios from 'axios'

export default {
  name: 'ChatBot',
  data() {
    return {
      messages: [{
        role: 'assistant',
        content: '您好！我是销售顾问张先生，很高兴为您服务。请问您想了解哪些二手车信息呢？'
      }],
      userInput: '',
      isProcessing: false,
      isTyping: false
    }
  },
  methods: {
    async sendMessage() {
      if (!this.userInput.trim() || this.isProcessing) return

      const userMessage = this.userInput.trim()
      this.messages.push({ role: 'user', content: userMessage })
      this.userInput = ''
      this.isProcessing = true
      this.isTyping = true

      try {
        const response = await axios.post('/api/chat', {
          messages: [{ role: 'user', content: userMessage }]
        })

        this.isTyping = false
        if (response.data.error) {
          this.messages.push({
            role: 'assistant',
            content: `抱歉，出现了一个错误：${response.data.error}`
          })
        } else {
          this.messages.push({
            role: 'assistant',
            content: response.data.response
          })
        }
      } catch (error) {
        this.isTyping = false
        this.messages.push({
          role: 'assistant',
          content: '抱歉，服务器连接失败，请稍后再试。'
        })
        console.error('Error:', error)
      }

      this.isProcessing = false
      this.$nextTick(() => {
        this.scrollToBottom()
      })
    },
    scrollToBottom() {
      const chatMessages = this.$refs.chatMessages
      chatMessages.scrollTop = chatMessages.scrollHeight
    }
  },
  mounted() {
    this.checkServerHealth()
  }
}
</script> 