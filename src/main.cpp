#include <WiFi.h>
#include <Wire.h>
#include <OV7670.h>
#include "esp_http_server.h"
#include "gesture_model.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


// WiFi SoftAP
const char* AP_SSID = "ESP32-Gesture";
const char* AP_PASS = "12345678";

// Пины камеры
#define PIN_D0     36
#define PIN_D1     39
#define PIN_D2     34
#define PIN_D3     35
#define PIN_D4     16
#define PIN_D5     17
#define PIN_D6     25
#define PIN_D7     26
#define PIN_XCLK   27
#define PIN_PCLK   14
#define PIN_VSYNC  13
#define PIN_SDA    21
#define PIN_SCL    22

// Размеры
#define CAM_W      160
#define CAM_H      120
#define NN_W       32
#define NN_H       32
#define BUF_SIZE   (CAM_W * CAM_H * 2)  // 38400 байт RGB565
#define NN_SIZE    (NN_W * NN_H)         // 1024 байт grayscale

// Классы (русские названия)
const char* CLASS_NAMES[9] = {
    "ничего нет",           // 0: none
    "один палец",           // 1: one_finger
    "два пальца",           // 2: two_fingers
    "три пальца",           // 3: three_fingers
    "четыре пальца",        // 4: four_fingers
    "пять пальцев",         // 5: five_fingers
    "круг",                 // 6: circle
    "квадрат",              // 7: square
    "треугольник"           // 8: triangle
};

// ==================== ГЛОБАЛЬНЫЕ ОБЪЕКТЫ ====================

OV7670 camera;
camera_config_t cam_config;
uint8_t* frame_buffer = nullptr;

// Результат классификации
float class_confidences[9] = {0};
int   best_class = 0;
float best_conf  = 0.0f;
SemaphoreHandle_t result_mutex = nullptr;

// TFLite
tflite::MicroInterpreter* interpreter = nullptr;
tflite::AllOpsResolver   resolver;
const tflite::Model*      model = nullptr;

constexpr int kTensorArenaSize = 90 * 1024;
uint8_t* tensor_arena = nullptr;

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

inline uint8_t rgb565_to_gray(uint16_t rgb) {
    uint8_t r = (rgb >> 11) & 0x1F;
    uint8_t g = (rgb >> 5)  & 0x3F;
    uint8_t b =  rgb        & 0x1F;
    return (uint8_t)(((uint16_t)r * 77 + (uint16_t)g * 150 + (uint16_t)b * 29) >> 8);
}

// Даунсэмплинг 160x120 RGB565 → 32x32 grayscale
void downsample_grayscale(const uint8_t* rgb565, uint8_t* gray_out) {
    for (int y = 0; y < NN_H; y++) {
        int src_y = (y * CAM_H) / NN_H;
        const uint16_t* row = (const uint16_t*)(rgb565 + src_y * CAM_W * 2);
        for (int x = 0; x < NN_W; x++) {
            int src_x = (x * CAM_W) / NN_W;
            gray_out[y * NN_W + x] = rgb565_to_gray(row[src_x]);
        }
    }
}

inline int8_t gray_to_int8(uint8_t gray) {
    return (int8_t)((int16_t)gray - 128);
}

void run_inference(uint8_t* gray_buf) {
    if (!interpreter) return;
    int8_t* input = interpreter->input(0)->data.int8;
    for (int i = 0; i < NN_SIZE; i++) input[i] = gray_to_int8(gray_buf[i]);
    if (interpreter->Invoke() != kTfLiteOk) return;
    
    int8_t* output = interpreter->output(0)->data.int8;
    float output_scale = interpreter->output(0)->params.scale;
    int   output_zero  = interpreter->output(0)->params.zero_point;
    
    float max_conf = -999.0f;
    int   max_idx  = 0;
    for (int i = 0; i < 9; i++) {
        float conf = (output[i] - output_zero) * output_scale;
        class_confidences[i] = conf;
        if (conf > max_conf) { max_conf = conf; max_idx = i; }
    }
    if (xSemaphoreTake(result_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        best_class = max_idx;
        best_conf  = max_conf;
        xSemaphoreGive(result_mutex);
    }
}

static uint8_t last_gray_buf[NN_SIZE];

// ==================== HTTP СЕРВЕР ====================

static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gesture Recognition + Capture PNG</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;display:flex}
/* === LEFT PANEL === */
#left{flex:0 0 480px;display:flex;flex-direction:column;align-items:center;padding:8px;overflow-y:auto;border-right:1px solid #0f3460}
h2{color:#00d4aa;margin:4px 0;font-size:1em}
.panel{background:#16213e;border-radius:10px;padding:8px;border:1px solid #0f3460}
.panel h3{color:#00d4aa;font-size:0.75em;margin-bottom:4px}
canvas{border-radius:6px;image-rendering:pixelated}
#nnView{width:200px;image-rendering:pixelated}
#resultBox{background:#16213e;border-radius:10px;padding:10px;margin:6px 0;width:100%;max-width:460px;border:2px solid #0f3460}
#className{font-size:1.5em;font-weight:bold;color:#00d4aa;margin:4px 0}
#confidence{font-size:1em;color:#e94560}
.bars{margin:6px 0;text-align:left}
.bar-row{display:flex;align-items:center;margin:2px 0;font-size:0.7em}
.bar-label{width:95px;text-align:right;padding-right:6px;color:#aaa}
.bar-track{flex:1;background:#0f3460;border-radius:3px;height:14px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#00d4aa,#00ff88);border-radius:3px;transition:width 0.15s}
.bar-pct{width:44px;text-align:right;padding-left:4px;color:#00d4aa;font-size:0.9em}
button{background:#00d4aa;color:#000;border:none;padding:6px 12px;margin:3px;border-radius:6px;cursor:pointer;font-weight:bold;font-size:0.8em}
button:hover{background:#00ff88}
.capture-bar{background:#0f3460;margin:6px 0;padding:8px;border-radius:10px;width:100%;max-width:460px}
.capture-bar select,.capture-bar button{margin:0 4px;padding:4px 8px;font-size:0.75em}
.capture-bar span{font-size:0.75em}
#fps{color:#666;font-size:0.7em;margin-top:2px}
/* === RIGHT PANEL (GUIDE) === */
#right{flex:1;overflow-y:auto;padding:10px 14px;min-width:340px}
#right h1{color:#00d4aa;font-size:1.1em;margin:8px 0;text-align:center}
#right h2{color:#00d4aa;font-size:0.9em;margin:14px 0 6px;border-bottom:1px solid #0f3460;padding-bottom:2px}
#right h3{color:#00ff88;font-size:0.8em;margin:10px 0 4px}
#right p,#right li{line-height:1.5;margin:2px 0;font-size:0.75em}
#right ul,#right ol{padding-left:16px;margin:4px 0}
#right code{background:#0f3460;padding:0 4px;border-radius:2px;color:#00d4aa;font-size:0.85em}
#right pre{background:#0f3460;padding:6px 8px;border-radius:6px;overflow-x:auto;font-size:0.68em;margin:4px 0;line-height:1.35}
#right pre code{background:none;padding:0;color:#e0e0e0}
#right .gpanel{background:#16213e;border-radius:10px;padding:8px 10px;border:1px solid #0f3460;margin:6px 0}
#right table{width:100%;border-collapse:collapse;margin:4px 0;font-size:0.72em}
#right th,#right td{border:1px solid #0f3460;padding:3px 6px;text-align:left}
#right th{background:#0f3460;color:#00d4aa}
#right .note{background:#16213e;border-left:3px solid #e94560;padding:6px 10px;border-radius:0 6px 6px 0;margin:6px 0;font-size:0.72em}
#right .warn{background:#16213e;border-left:3px solid #ffaa00;padding:6px 10px;border-radius:0 6px 6px 0;margin:6px 0;font-size:0.72em}
#right .step{color:#ffaa00;font-weight:bold}
#right strong{color:#fff}
/* === RESPONSIVE === */
@media(max-width:820px){
  body{flex-direction:column}
  #left{flex:0 0 auto;width:100%;border-right:none;border-bottom:1px solid #0f3460}
  #right{flex:1;min-width:0}
}
</style>
</head><body>
<!-- ========== LEFT PANEL ========== -->
<div id="left">
<h2>OV7670 32x32 + ESP32 ИИ распознавание жестов</h2>
<div class="panel" style="margin:4px 0"><h3>Neural Net Input (32x32)</h3><canvas id="nnView" width="128" height="128"></canvas></div>
<div id="resultBox">
  <div id="className">---</div>
  <div id="confidence"></div>
  <div class="bars" id="bars"></div>
</div>
<div class="capture-bar">
  <span>📸 Класс:</span>
  <select id="classSelect">
    <option value="0">0 - ничего нет</option>
    <option value="1">1 - один палец</option>
    <option value="2">2 - два пальца</option>
    <option value="3">3 - три пальца</option>
    <option value="4">4 - четыре пальца</option>
    <option value="5">5 - пять пальцев</option>
    <option value="6">6 - круг</option>
    <option value="7">7 - квадрат</option>
    <option value="8">8 - треугольник</option>
  </select>
  <button id="captureBtn">📸 Снять PNG (Пробел)</button>
</div>
<div id="fps"></div>
</div>

<!-- ========== RIGHT PANEL (GUIDE) ========== -->
<div id="right">
<h1>OV7670 + ESP32 + TFLite — распознавание жестов</h1>

<div class="gpanel">
<h2>Что делает проект</h2>
<p>Распознавание 9 жестов руки в реальном времени: камера <code>OV7670</code> (160×120 RGB565) → даунсэмпл 32×32 grayscale → нейросеть TFLite Micro на ESP32 → результат через WiFi веб-интерфейс.</p>
<p><strong>9 классов:</strong> ничего нет, 1–5 пальцев, круг, квадрат, треугольник.</p>
</div>

<h2>Что нужно</h2>
<table>
<tr><th>Компонент</th><th>Примечание</th></tr>
<tr><td>ESP32</td><td>WEMOS D1 UNO32</td></tr>
<tr><td>OV7670 <strong>без FIFO</strong></td><td>Захват через I2S</td></tr>
<tr><td>Провода (~20 шт.)</td><td>Короткие (PCLK, XCLK)</td></tr>
<tr><td>VSCode + PlatformIO</td><td></td></tr>
<tr><td>Python 3.8+</td><td>Обучение модели</td></tr>
</table>

<h2>Подключение пинов OV7670 → ESP32</h2>
<table>
<tr><th>OV7670</th><th>GPIO</th><th>Зачем</th></tr>
<tr><td>3.3V</td><td>3.3V</td><td>Питание (до 150 мА)</td></tr>
<tr><td>GND</td><td>GND</td><td></td></tr>
<tr><td>SIOC</td><td>22</td><td>I2C SCL</td></tr>
<tr><td>SIOD</td><td>21</td><td>I2C SDA</td></tr>
<tr><td>VSYNC</td><td>13</td><td>Прерывание (neg. edge)</td></tr>
<tr><td>PCLK</td><td>14</td><td>Pixel Clock (I2S)</td></tr>
<tr><td>XCLK</td><td>27</td><td>Такт 8 МГц (LEDC)</td></tr>
<tr><td>D0</td><td>36</td><td>Данные (input-only)</td></tr>
<tr><td>D1</td><td>39</td><td>Данные (input-only)</td></tr>
<tr><td>D2</td><td>34</td><td>Данные</td></tr>
<tr><td>D3</td><td>35</td><td>Данные</td></tr>
<tr><td>D4</td><td>16</td><td>Данные</td></tr>
<tr><td>D5</td><td>17</td><td>Данные</td></tr>
<tr><td>D6</td><td>25</td><td>Данные</td></tr>
<tr><td>D7</td><td>26</td><td>Данные</td></tr>
</table>
<div class="note"><strong>Важно:</strong> HREF не используется — постоянный HIGH. GPIO 36 и 39 — <strong>только вход</strong>. I2C на 50 кГц.</div>

<h2>Как собрать и загрузить</h2>
<ol>
<li>Установить <strong>PlatformIO</strong> (<code>pip install platformio</code>)</li>
<li>Структура проекта:</li>
</ol>
<pre><code>AI-esp32/
├── platformio.ini
├── src/main.cpp
├── include/gesture_model.h
└── lib/OV7670/
    ├── OV7670.h / OV7670.cpp
    └── I2Scamera.h / I2Scamera.c</code></pre>
<ol start="3">
<li>Подключить ESP32 по USB:</li>
</ol>
<pre><code>pio run --target upload
pio device monitor --baud 115200</code></pre>
<ol start="4">
<li>WiFi <code>ESP32-Gesture</code> (пароль <code>12345678</code>), открыть <code>http://192.168.4.1</code></li>
</ol>

<h2>Обучение модели (4 шага)</h2>

<h3><span class="step">Шаг 1:</span> Сбор seed-фото 32×32</h3>
<p>Через кнопку «Снять PNG» (слева) сохраняются PNG с реальным шумом камеры.</p>
<pre><code>real_seeds/
├── none/  one_finger/  two_fingers/
├── three_fingers/  four_fingers/  five_fingers/
├── circle/  square/  triangle/</code></pre>

<h3><span class="step">Шаг 2:</span> Аугментация</h3>
<p>Скрипт <code>augment_seeds.py</code>: поворот ±8°, сдвиг ±2px, масштаб 0.95–1.05, яркость/контраст, flip. <strong>Без</strong> искусственного шума.</p>
<p>2000 img/класс, ×2 для сложных. Дедупликация по MD5.</p>

<h3><span class="step">Шаг 3:</span> Обучение (main.py)</h3>
<table>
<tr><th>Слой</th><th>Параметры</th></tr>
<tr><td>Conv2D+BN+ReLU</td><td>32×3×3</td></tr>
<tr><td>MaxPool</td><td>2×2</td></tr>
<tr><td>Conv2D+BN+ReLU</td><td>64×3×3</td></tr>
<tr><td>MaxPool</td><td>→ 8×8×64</td></tr>
<tr><td>Conv2D+BN+ReLU</td><td>128×3×3</td></tr>
<tr><td>GlobalAvgPool</td><td>→ 128</td></tr>
<tr><td>Dense+ReLU</td><td>64</td></tr>
<tr><td>Dropout 0.3</td><td></td></tr>
<tr><td>Dense+Softmax</td><td>9 классов</td></tr>
</table>
<p>~100K параметров. Adam lr=0.001, EarlyStopping(7), ReduceLROnPlateau, 70/15/15 split.</p>

<h3><span class="step">Шаг 4:</span> Конвертация в TFLite Int8</h3>
<p>Автоматически в <code>main.py</code>. Representative dataset (100 сэмплов), int8 вход/выход. ~100 КБ.</p>

<h2>Важные советы</h2>
<div class="gpanel">
<ul>
<li><strong>Питание:</strong> OV7670 до 150 мА. Внешний стабилизатор 3.3V.</li>
<li><strong>Провода:</strong> короче — лучше. PCLK и XCLK критичны.</li>
<li><strong>Память:</strong> ~90 КБ тензорной арены. <code>kTensorArenaSize</code>.</li>
<li><strong>FPS:</strong> ~15-20 кадров/с. Инференс ~30-50 мс.</li>
<li><strong>Плата:</strong> <code>wemos_d1_uno32</code>. Менять <code>board</code> в platformio.ini.</li>
<li><strong>Модель:</strong> int8 32×32×1 вход, 9 классов выход.</li>
<li><strong>I2C-адрес:</strong> <code>0x21</code>.</li>
<li><strong>Seed vs синтетика:</strong> реальные фото с камеры критически улучшают качество.</li>
<li><strong>Баланс классов:</strong> равное количество фото. Class weights помогают.</li>
<li><strong>Нормализация:</strong> <code>(x/255-0.5)*2</code> → [-1,1]. ESP32: <code>gray-128</code>.</li>
</ul>
</div>
<p style="text-align:center;color:#666;margin-top:14px;font-size:0.68em">ESP32 Gesture Recognition Project</p>
</div>

<script>
let autoSpeak=true, lastClass=-1, fps=0, lastFps=Date.now(), frameCount=0;
const cn=['ничего нет','один палец','два пальца','три пальца','четыре пальца','пять пальцев','круг','квадрат','треугольник'];
const nnCtx=document.getElementById('nnView').getContext('2d');
const nnImg=new ImageData(32,32);
const tmpCan=document.createElement('canvas'); tmpCan.width=32; tmpCan.height=32;
const tmpCtx=tmpCan.getContext('2d');

async function update(){
  try{
    let t0=Date.now();
    let r=await fetch('/frame?t='+t0);
    if(!r.ok){setTimeout(update,30);return}
    let buf=await r.arrayBuffer();
    if(buf.byteLength<1068){setTimeout(update,30);return}
    let dv=new DataView(buf), off=0;
    let cls=dv.getInt32(off,true); off+=4;
    let cf=dv.getInt32(off,true)/10000; off+=4;
    let probs=[];
    for(let i=0;i<9;i++){probs.push(dv.getFloat32(off,true));off+=4;}
    let gray=new Uint8Array(buf,off,1024);
    for(let i=0;i<1024;i++){
      let v=gray[i];
      nnImg.data[i*4]=v; nnImg.data[i*4+1]=v; nnImg.data[i*4+2]=v; nnImg.data[i*4+3]=255;
    }
    tmpCtx.putImageData(nnImg,0,0);
    nnCtx.imageSmoothingEnabled=false;
    nnCtx.drawImage(tmpCan,0,0,128,128);
    document.getElementById('className').textContent=cn[cls]||'???';
    document.getElementById('confidence').textContent='Уверенность: '+(cf*100).toFixed(1)+'%';
    let bars='';
    for(let i=0;i<9;i++){
      let pct=Math.max(0,probs[i]*100);
      let color=pct>50?'#00ff88':pct>20?'#ffaa00':'#e94560';
      bars+='<div class="bar-row"><div class="bar-label">'+cn[i]+'</div><div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div><div class="bar-pct">'+pct.toFixed(1)+'%</div></div>';
    }
    document.getElementById('bars').innerHTML=bars;
    if(autoSpeak&&cls!==lastClass&&cf>0.5){lastClass=cls;speakNow(cn[cls]);}
    frameCount++;
    let now=Date.now();
    if(now-lastFps>=1000){fps=frameCount;frameCount=0;lastFps=now;}
    document.getElementById('fps').textContent='FPS: '+fps+' ('+(Date.now()-t0)+'ms)';
  }catch(e){console.error(e)}
  setTimeout(update,30);
}

async function saveAsPNG() {
  let classId = document.getElementById('classSelect').value;
  let now = new Date();
  let timestamp = now.getFullYear() + '' + (now.getMonth()+1).toString().padStart(2,'0') + '' + now.getDate().toString().padStart(2,'0') + '_' +
                  now.getHours().toString().padStart(2,'0') + now.getMinutes().toString().padStart(2,'0') + now.getSeconds().toString().padStart(2,'0');
  let filename = `class_${classId}_${timestamp}.png`;
  
  try {
    let resp = await fetch(`/capture_nn?class=${classId}`);
    if (!resp.ok) throw new Error('HTTP error');
    let grayData = new Uint8Array(await resp.arrayBuffer());
    if (grayData.length !== 1024) throw new Error('Invalid data size');
    
    let canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    let ctx = canvas.getContext('2d');
    let imgData = ctx.createImageData(32, 32);
    for (let i = 0; i < 1024; i++) {
      let v = grayData[i];
      imgData.data[i*4] = v;
      imgData.data[i*4+1] = v;
      imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    
    canvas.toBlob((blob) => {
      let link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
      document.getElementById('fps').innerHTML += ' ✓ PNG saved';
      setTimeout(()=>{document.getElementById('fps').innerHTML = 'FPS: '+fps;}, 1000);
    }, 'image/png');
    
  } catch(e) {
    console.error(e);
    alert('Save failed: ' + e);
  }
}

function speakNow(text){
  if(!window.speechSynthesis)return;
  speechSynthesis.cancel();
  let u=new SpeechSynthesisUtterance(text);
  u.lang='ru-RU';u.rate=0.9;u.pitch=1.0;
  speechSynthesis.speak(u);
}

document.getElementById('captureBtn').onclick = saveAsPNG;
window.addEventListener('keydown', (e) => { if(e.code === 'Space') { e.preventDefault(); saveAsPNG(); } });
update();
</script>
</body></html>
)rawliteral";

static const char GUIDE_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Гайд — OV7670 + ESP32 + TFLite</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;padding:10px;max-width:800px;margin:0 auto}
h1{color:#00d4aa;font-size:1.4em;margin:12px 0;text-align:center}
h2{color:#00d4aa;font-size:1.1em;margin:20px 0 10px;border-bottom:1px solid #0f3460;padding-bottom:4px}
h3{color:#00ff88;font-size:1em;margin:14px 0 6px}
p,li{line-height:1.6;margin:4px 0;font-size:0.9em}
ul,ol{padding-left:20px;margin:6px 0}
code{background:#0f3460;padding:1px 5px;border-radius:3px;color:#00d4aa;font-size:0.85em}
pre{background:#0f3460;padding:10px;border-radius:8px;overflow-x:auto;font-size:0.78em;margin:8px 0;line-height:1.4}
pre code{background:none;padding:0;color:#e0e0e0}
.panel{background:#16213e;border-radius:12px;padding:12px;border:1px solid #0f3460;margin:10px 0}
table{width:100%;border-collapse:collapse;margin:8px 0;font-size:0.85em}
th,td{border:1px solid #0f3460;padding:6px 10px;text-align:left}
th{background:#0f3460;color:#00d4aa}
td code{white-space:nowrap}
.note{background:#16213e;border-left:3px solid #e94560;padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0;font-size:0.85em}
.warn{background:#16213e;border-left:3px solid #ffaa00;padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0;font-size:0.85em}
.back{display:inline-block;background:#00d4aa;color:#000;padding:6px 14px;border-radius:6px;text-decoration:none;font-weight:bold;margin:10px 0;font-size:0.85em}
.back:hover{background:#00ff88}
.step{color:#ffaa00;font-weight:bold}
strong{color:#fff}
</style>
</head><body>
<a href="/" class="back">← На главную</a>
<h1>OV7670 + ESP32 + TFLite — распознавание жестов</h1>

<div class="panel">
<h2>Что делает проект</h2>
<p>Распознавание 9 жестов руки в реальном времени: камера <code>OV7670</code> (160×120 RGB565) → даунсэмпл 32×32 grayscale → нейросеть TFLite Micro на ESP32 → результат через WiFi веб-интерфейс.</p>
<p><strong>9 классов:</strong> ничего нет, 1–5 пальцев, круг, квадрат, треугольник.</p>
</div>

<h2>Что нужно</h2>
<table>
<tr><th>Компонент</th><th>Примечание</th></tr>
<tr><td>ESP32 (любая плата)</td><td>WEMOS D1 UNO32 в проекте</td></tr>
<tr><td>OV7670 <strong>без FIFO</strong></td><td>Захват через I2S, версия с AL422B не подойдёт</td></tr>
<tr><td>Провода (~20 шт.)</td><td>Желательно короткие (особенно для PCLK, XCLK)</td></tr>
<tr><td>VSCode + PlatformIO</td><td></td></tr>
<tr><td>Python 3.8+</td><td>Только для обучения модели</td></tr>
</table>

<h2>Подключение пинов OV7670 → ESP32</h2>
<table>
<tr><th>OV7670</th><th>ESP32 GPIO</th><th>Зачем</th></tr>
<tr><td>3.3V</td><td>3.3V</td><td>Питание (камера до 150 мА)</td></tr>
<tr><td>GND</td><td>GND</td><td></td></tr>
<tr><td>SIOC</td><td>22</td><td>I2C SCL (настройка регистров)</td></tr>
<tr><td>SIOD</td><td>21</td><td>I2C SDA</td></tr>
<tr><td>VSYNC</td><td>13</td><td>Прерывание по спаду (neg. edge)</td></tr>
<tr><td>PCLK</td><td>14</td><td>Pixel Clock (через I2S)</td></tr>
<tr><td>XCLK</td><td>27</td><td>Тактовый сигнал 8 МГц (LEDC)</td></tr>
<tr><td>D0</td><td>36</td><td>Данные (input-only)</td></tr>
<tr><td>D1</td><td>39</td><td>Данные (input-only)</td></tr>
<tr><td>D2</td><td>34</td><td>Данные</td></tr>
<tr><td>D3</td><td>35</td><td>Данные</td></tr>
<tr><td>D4</td><td>16</td><td>Данные</td></tr>
<tr><td>D5</td><td>17</td><td>Данные</td></tr>
<tr><td>D6</td><td>25</td><td>Данные</td></tr>
<tr><td>D7</td><td>26</td><td>Данные</td></tr>
</table>
<div class="note"><strong>Важно:</strong> HREF не используется — вместо него подан постоянный высокий уровень. GPIO 36 и 39 — <strong>только вход</strong>, без подтяжек. I2C скорость снижена до 50 кГц для стабильности.</div>

<h2>Как собрать и загрузить</h2>
<ol>
<li>Установить <strong>PlatformIO</strong> (расширение VSCode или <code>pip install platformio</code>)</li>
<li>Клонировать/создать проект, скопировать все файлы по структуре:</li>
</ol>
<pre><code>AI-esp32/
├── platformio.ini
├── src/main.cpp
├── include/gesture_model.h
└── lib/OV7670/
    ├── OV7670.h / OV7670.cpp
    └── I2Scamera.h / I2Scamera.c</code></pre>
<ol start="3">
<li>Подключить ESP32 по USB, выполнить:</li>
</ol>
<pre><code>pio run --target upload
pio device monitor --baud 115200</code></pre>
<ol start="4">
<li>Подключиться к WiFi <code>ESP32-Gesture</code> (пароль <code>12345678</code>), открыть <code>http://192.168.4.1</code></li>
</ol>

<h2>Как обучить свою модель</h2>
<p>Обучение модели — самая важная часть проекта. Весь процесс состоит из 4 шагов: сбор seed-фото, аугментация, обучение, конвертация в TFLite.</p>

<h3><span class="step">Шаг 1:</span> Сбор seed-фотографий (реальные снимки 32×32)</h3>
<p><strong>Сбор с ESP32 напрямую:</strong> через веб-интерфейс проекта (кнопка «Снять PNG» или пробел) сохраняются PNG 32×32 — это уже готовые seed-изображения с реальным шумом камеры OV7670.</p>
<p><strong>Структура seed-фото:</strong></p>
<pre><code>real_seeds/
├── none/           # пустой фон
├── one_finger/     # 1 палец
├── two_fingers/    # 2 пальца
├── three_fingers/  # 3 пальца
├── four_fingers/   # 4 пальца
├── five_fingers/   # 5 пальцев
├── circle/         # жест «ОК»
├── square/         # квадрат из пальцев
├── triangle/       # треугольник из пальцев</code></pre>

<h3><span class="step">Шаг 2:</span> Аугментация seed-фото</h3>
<p>Скрипт <code>augment_seeds.py</code> берёт реальные seed-фото из <code>real_seeds/</code> и генерирует датасет с мягкими аугментациями.</p>
<p><strong>Применяемые аугментации (только безопасные):</strong></p>
<ul>
<li>Поворот до ±8° (70% вероятность)</li>
<li>Сдвиг до ±2 px (70%)</li>
<li>Масштаб 0.95–1.05 (50%)</li>
<li>Случайный crop с отражением краёв (50%)</li>
<li>Изменение яркости/контраста &alpha; 0.9–1.1, &beta; −10..10 (50%)</li>
<li>Горизонтальное отражение (30%)</li>
</ul>
<div class="warn"><strong>Важно:</strong> скрипт НЕ добавляет шум, размытие, JPEG-артефакты — они уже есть в seed-фото.</div>
<p><strong>Настройки:</strong> 2000 изображений на класс. Для сложных классов (<code>two_fingers</code>, <code>three_fingers</code>) — ×2 (4000). Отсеиваются дубликаты по MD5-хешу.</p>

<h3><span class="step">Шаг 3:</span> Обучение модели</h3>
<p>Скрипт <code>main.py</code> — полный пайплайн: загрузка датасета → обучение → оценка → конвертация в TFLite.</p>
<p><strong>Архитектура модели (CNN для 32×32×1):</strong></p>
<table>
<tr><th>Слой</th><th>Параметры</th></tr>
<tr><td>Conv2D + BN + ReLU</td><td>32 фильтра 3×3, same</td></tr>
<tr><td>MaxPooling2D</td><td>2×2</td></tr>
<tr><td>Conv2D + BN + ReLU</td><td>64 фильтра 3×3, same</td></tr>
<tr><td>MaxPooling2D</td><td>2×2 → 8×8×64</td></tr>
<tr><td>Conv2D + BN + ReLU</td><td>128 фильтров 3×3, same</td></tr>
<tr><td>GlobalAveragePooling2D</td><td>→ 128</td></tr>
<tr><td>Dense + ReLU</td><td>64 нейрона</td></tr>
<tr><td>Dropout</td><td>0.3</td></tr>
<tr><td>Dense + Softmax</td><td>9 классов</td></tr>
</table>
<p><strong>Почему именно такая архитектура:</strong></p>
<ul>
<li>BatchNormalization после каждого Conv2D — критично для стабильности при int8-квантизации</li>
<li>GlobalAveragePooling вместо Flatten — меньше параметров, устойчивее к сдвигам</li>
<li>Всего ~100K параметров — помещается в память ESP32</li>
<li>Вход: float32 [-1, 1] при обучении, int8 [-128, 127] на ESP32</li>
</ul>
<p><strong>Процесс обучения:</strong></p>
<ul>
<li>Разделение: 70% train / 15% val / 15% test (stratify по классам)</li>
<li>Class weights для балансировки</li>
<li>Оптимизатор Adam (lr=0.001)</li>
<li>EarlyStopping (patience=7, monitor=val_accuracy)</li>
<li>ReduceLROnPlateau (factor 0.5, patience=3)</li>
<li>ModelCheckpoint — сохраняет лучшую модель в <code>best_model.h5</code></li>
<li>Эпох: до 50 (обычно early stopping на 20–30)</li>
</ul>

<h3><span class="step">Шаг 4:</span> Конвертация в TFLite Int8</h3>
<p>Конвертация происходит автоматически в <code>main.py</code>.</p>
<p><strong>Ключевые моменты:</strong></p>
<ul>
<li><strong>Representative dataset</strong> — 100 сэмплов из валидационной выборки для калибровки квантизации</li>
<li><strong>Int8 вход и выход</strong> — обязательно для совместимости с ESP32 (TFLite Micro)</li>
<li><strong>Оптимизация DEFAULT</strong> — включает квантизацию весов и активаций</li>
<li>Размер модели: ~100 КБ</li>
</ul>

<h2>Важные советы</h2>
<div class="panel">
<ul>
<li><strong>Питание:</strong> OV7670 потребляет до 150 мА. При нестабильном питании камера не инициализируется. Используйте внешний стабилизатор 3.3V.</li>
<li><strong>Провода:</strong> чем короче, тем лучше. PCLK и XCLK — самые критичные.</li>
<li><strong>Память:</strong> модель требует ~90 КБ тензорной арены. Если <code>AllocateTensors</code> падает — уменьшите разрешение или увеличьте <code>kTensorArenaSize</code>.</li>
<li><strong>FPS:</strong> ~15-20 кадров/с. Инференс — самая медленная часть (~30-50 мс).</li>
<li><strong>Плата:</strong> проект настроен на <code>wemos_d1_uno32</code>. Для других плат поменять <code>board</code> в <code>platformio.ini</code>.</li>
<li><strong>Модель:</strong> вход int8 32×32×1, выход 9 классов. Квантизация int8 обязательна.</li>
<li><strong>I2C-адрес камеры:</strong> <code>0x21</code>. Если камера не отвечает — проверьте SDA/SCL и питание.</li>
<li><strong>Seed-фото vs синтетика:</strong> синтетический датасет даёт базовую точность. Реальные seed-фото с камеры критически улучшают качество.</li>
<li><strong>Баланс классов:</strong> собирайте примерно равное количество фото на каждый класс. Class weights помогают, но не компенсируют сильный дисбаланс.</li>
<li><strong>Нормализация:</strong> при обучении данные нормализуются в <code>(x/255 - 0.5) * 2</code> → [-1, 1]. На ESP32 вход int8: <code>gray - 128</code>.</li>
</ul>
</div>

<a href="/" class="back">← На главную</a>
<p style="text-align:center;color:#666;margin-top:20px;font-size:0.75em">ESP32 Gesture Recognition Project</p>
</body></html>
)rawliteral";

httpd_handle_t http_server = NULL;

static esp_err_t handle_index(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_send(req, INDEX_HTML, strlen_P(INDEX_HTML));
    return ESP_OK;
}

static esp_err_t handle_raw(httpd_req_t *req) {
    if (!frame_buffer) { httpd_resp_send_500(req); return ESP_FAIL; }
    camera.getFrame(frame_buffer);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_send(req, (const char*)frame_buffer, BUF_SIZE);
    return ESP_OK;
}

static esp_err_t handle_nn(httpd_req_t *req) {
    if (!frame_buffer) { httpd_resp_send_500(req); return ESP_FAIL; }
    uint8_t gray_buf[NN_SIZE];
    camera.getFrame(frame_buffer);
    downsample_grayscale(frame_buffer, gray_buf);
    run_inference(gray_buf);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_send(req, (const char*)gray_buf, NN_SIZE);
    return ESP_OK;
}

static esp_err_t handle_result(httpd_req_t *req) {
    int cls = 0; float conf = 0.0f; float probs[9] = {0};
    if (xSemaphoreTake(result_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        cls = best_class; conf = best_conf; memcpy(probs, class_confidences, sizeof(probs));
        xSemaphoreGive(result_mutex);
    }
    char json[1024];
    int len = snprintf(json, sizeof(json), "{\"class\":%d,\"conf\":%.4f,\"probs\":[", cls, conf);
    for (int i = 0; i < 9; i++) len += snprintf(json+len, sizeof(json)-len, "%.4f%s", probs[i], (i<8)?",":"");
    len += snprintf(json+len, sizeof(json)-len, "]}");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, len);
    return ESP_OK;
}

static esp_err_t handle_frame(httpd_req_t *req) {
    if (!frame_buffer) { httpd_resp_send_500(req); return ESP_FAIL; }
    camera.getFrame(frame_buffer);
    downsample_grayscale(frame_buffer, last_gray_buf);
    run_inference(last_gray_buf);
    int cls = 0; float conf = 0.0f; float probs[9] = {0};
    if (xSemaphoreTake(result_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        cls = best_class; conf = best_conf; memcpy(probs, class_confidences, sizeof(probs));
        xSemaphoreGive(result_mutex);
    }
    int32_t cls_le = cls, conf_le = (int32_t)(conf * 10000.0f);
    uint8_t hdr[44];
    memcpy(hdr, &cls_le, 4); memcpy(hdr+4, &conf_le, 4); memcpy(hdr+8, probs, 36);
    uint8_t payload[44 + NN_SIZE];
    memcpy(payload, hdr, 44); memcpy(payload+44, last_gray_buf, NN_SIZE);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_send(req, (const char*)payload, sizeof(payload));
    return ESP_OK;
}

// Эндпоинт для отображения гайда
static esp_err_t handle_guide(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_send(req, GUIDE_HTML, strlen_P(GUIDE_HTML));
    return ESP_OK;
}

// Эндпоинт возвращает только 1024 байта grayscale 32x32
static esp_err_t handle_capture_nn(httpd_req_t *req) {
    if (!frame_buffer) { httpd_resp_send_500(req); return ESP_FAIL; }
    camera.getFrame(frame_buffer);
    uint8_t gray_buf[NN_SIZE];
    downsample_grayscale(frame_buffer, gray_buf);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char*)gray_buf, NN_SIZE);
    return ESP_OK;
}

static void start_http_server() {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.max_uri_handlers = 9;
    cfg.lru_purge_enable = true;
    if (httpd_start(&http_server, &cfg) == ESP_OK) {
        httpd_uri_t uri_idx   = { .uri = "/", .method = HTTP_GET, .handler = handle_index };
        httpd_uri_t uri_raw   = { .uri = "/raw", .method = HTTP_GET, .handler = handle_raw };
        httpd_uri_t uri_nn    = { .uri = "/nn", .method = HTTP_GET, .handler = handle_nn };
        httpd_uri_t uri_res   = { .uri = "/result", .method = HTTP_GET, .handler = handle_result };
        httpd_uri_t uri_frame = { .uri = "/frame", .method = HTTP_GET, .handler = handle_frame };
        httpd_uri_t uri_capnn = { .uri = "/capture_nn", .method = HTTP_GET, .handler = handle_capture_nn };
        httpd_uri_t uri_guide = { .uri = "/guide", .method = HTTP_GET, .handler = handle_guide };
        httpd_register_uri_handler(http_server, &uri_idx);
        httpd_register_uri_handler(http_server, &uri_raw);
        httpd_register_uri_handler(http_server, &uri_nn);
        httpd_register_uri_handler(http_server, &uri_res);
        httpd_register_uri_handler(http_server, &uri_frame);
        httpd_register_uri_handler(http_server, &uri_capnn);
        httpd_register_uri_handler(http_server, &uri_guide);
        Serial.println("  HTTP server on :80");
        Serial.println("  Endpoints: / /raw /nn /result /frame /capture_nn /guide");
    } else {
        Serial.println("  HTTP server FAILED");
    }
}

// ==================== SETUP ====================

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=== OV7670 + TFLite Gesture Recognition + PNG Capture ===\n");
    
    // I2C
    Serial.println("[1/5] I2C Setup");
    gpio_set_pull_mode(GPIO_NUM_21, GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(GPIO_NUM_22, GPIO_PULLUP_ONLY);
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(50000);
    delay(300);
    
    // Camera
    Serial.println("[2/5] Camera Init");
    cam_config.D0  = PIN_D0; cam_config.D1  = PIN_D1; cam_config.D2  = PIN_D2; cam_config.D3  = PIN_D3;
    cam_config.D4  = PIN_D4; cam_config.D5  = PIN_D5; cam_config.D6  = PIN_D6; cam_config.D7  = PIN_D7;
    cam_config.XCLK = PIN_XCLK; cam_config.PCLK = PIN_PCLK; cam_config.VSYNC = PIN_VSYNC;
    cam_config.xclk_freq_hz = 8000000;
    cam_config.ledc_timer   = LEDC_TIMER_0;
    cam_config.ledc_channel = LEDC_CHANNEL_0;
    
    esp_err_t err = camera.init(&cam_config, QQVGA, RGB565);
    if (err != ESP_OK) { Serial.printf("Camera init FAILED: %d\n", err); return; }
    Serial.println("  Camera OK");
    
    camera.setBright(24);
    camera.setContrast(0x58);
    
    frame_buffer = (uint8_t*)malloc(BUF_SIZE);
    if (!frame_buffer) { Serial.println("No mem for frame buffer!"); return; }
    
    // TFLite
    Serial.println("[3/5] TensorFlow Lite Init");
    model = tflite::GetModel(gesture_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema version mismatch\n"); return;
    }
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (!tensor_arena) { Serial.println("No mem for tensor arena!"); return; }
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("Tensor allocation FAILED!"); return; }
    TfLiteTensor* input = interpreter->input(0);
    Serial.printf("  Model input: %dx%dx%d type=%d\n", input->dims->data[1], input->dims->data[2], input->dims->data[3], input->type);
    Serial.printf("  Input quantization: scale=%.6f zero=%d\n", input->params.scale, input->params.zero_point);
    Serial.println("  TFLite OK");
    
    result_mutex = xSemaphoreCreateMutex();
    
    // WiFi SoftAP
    Serial.println("[4/5] WiFi SoftAP");
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.printf("  AP: %s\n", AP_SSID);
    Serial.printf("  IP:  %s\n", WiFi.softAPIP().toString().c_str());
    Serial.println("  Password: " + String(AP_PASS));
    
    // HTTP
    Serial.println("[5/5] HTTP Server");
    start_http_server();
    
    // Прогрев
    Serial.println("  Warmup inference...");
    uint8_t gray_buf[NN_SIZE];
    memset(gray_buf, 128, NN_SIZE);
    camera.getFrame(frame_buffer);
    downsample_grayscale(frame_buffer, gray_buf);
    run_inference(gray_buf);
    Serial.printf("  Warmup done. Class=%d (%s) conf=%.2f\n", best_class, CLASS_NAMES[best_class], best_conf);
    
    Serial.println("\n=== READY ===");
    Serial.print("Connect to WiFi: "); Serial.println(AP_SSID);
    Serial.print("Then open: http://"); Serial.println(WiFi.softAPIP());
    Serial.println();
}

void loop() {
    static unsigned long last_stat = 0;
    if (millis() - last_stat > 10000) {
        last_stat = millis();
        Serial.printf("[STAT] Class=%d (%s) conf=%.2f heap=%u\n", best_class, CLASS_NAMES[best_class], best_conf, ESP.getFreeHeap());
    }
    delay(100);
}