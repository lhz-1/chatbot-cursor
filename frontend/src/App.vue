<template>
  <div class="app">
    <Header />
    <Hero />
    <div class="container">
      <Features />
      <CarList :cars="cars" />
      <ChatBot />
    </div>
    <Footer />
  </div>
</template>

<script>
import Header from './components/Header.vue'
import Hero from './components/Hero.vue'
import Features from './components/Features.vue'
import CarList from './components/CarList.vue'
import ChatBot from './components/ChatBot.vue'
import Footer from './components/Footer.vue'

export default {
  name: 'App',
  components: {
    Header,
    Hero,
    Features,
    CarList,
    ChatBot,
    Footer
  },
  data() {
    return {
      cars: {
        "奔驰C级": {"价格": "28-35万", "年份": "2020", "里程": "3万公里", "状况": "极好", "颜色": "银色"},
        "宝马3系": {"价格": "25-32万", "年份": "2019", "里程": "4.5万公里", "状况": "良好", "颜色": "黑色"},
        "奥迪A4L": {"价格": "22-30万", "年份": "2019", "里程": "5万公里", "状况": "良好", "颜色": "白色"},
        "大众帕萨特": {"价格": "15-20万", "年份": "2018", "里程": "6万公里", "状况": "良好", "颜色": "灰色"},
        "丰田凯美瑞": {"价格": "16-22万", "年份": "2019", "里程": "4万公里", "状况": "极好", "颜色": "红色"},
        "本田雅阁": {"价格": "18-24万", "年份": "2020", "里程": "3.8万公里", "状况": "极好", "颜色": "蓝色"}
      }
    }
  }
}
</script> 