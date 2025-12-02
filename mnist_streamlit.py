"""
MNIST 手寫數字辨識神經網路 - 擴展版本
Demo01 - 設計你的神經網路 (含前處理、指標分析、Streamlit UI)

這個腳本建立一個神經網路模型來辨識 MNIST 手寫數字資料集，包含：
  - 豐富的數據前處理步驟（正規化、標準化、數據增強、檢驗分割）
  - 詳細的訓練指標和性能分析（混淆矩陣、分類報告、逐類精度）
  - 完整的 Streamlit 互動式應用（含實時繪圖辨識和可視化分析）
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pickle
import os
from datetime import datetime

# ===== 超參數設定 =====
N1 = 20  # 第一隱藏層神經元數
N2 = 20  # 第二隱藏層神經元數
N3 = 20  # 第三隱藏層神經元數
BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 0.087
VALIDATION_SPLIT = 0.2


# ===== 資料前處理函式庫 =====
class MNISTPreprocessor:
    """MNIST 數據前處理類別"""
    
    @staticmethod
    def load_data():
        """加載原始 MNIST 數據"""
        print("📥 正在加載 MNIST 數據集...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(f"   訓練資料: {x_train.shape}, 測試資料: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def normalize(x):
        """
        正規化：將像素值縮放至 [0, 1]
        公式: x_normalized = x / 255
        """
        return x / 255.0
    
    @staticmethod
    def standardize(x):
        """
        標準化：將數據轉為 0 均值和單位方差
        公式: x_std = (x - mean) / std
        """
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std + 1e-8), mean, std
    
    @staticmethod
    def flatten(x):
        """將 2D 圖像 (28, 28) 轉為 1D 向量 (784,)"""
        n_samples = x.shape[0]
        return x.reshape(n_samples, -1)
    
    @staticmethod
    def one_hot_encode(y, num_classes=10):
        """轉換標籤為 one-hot 編碼"""
        return to_categorical(y, num_classes)
    
    @staticmethod
    def data_augmentation(x, intensity=0.1):
        """
        數據增強：添加小量高斯噪聲以增加模型魯棒性
        """
        noise = np.random.normal(0, intensity, x.shape)
        x_augmented = np.clip(x + noise, 0, 1)
        return x_augmented
    
    @staticmethod
    def preprocess_pipeline(x_train, y_train, x_test, y_test):
        """完整的前處理流程"""
        print("\n🔧 開始前處理...")
        
        # 步驟 1: 正規化
        print("  [1/6] 正規化像素值 (0-1)...")
        x_train_norm = MNISTPreprocessor.normalize(x_train)
        x_test_norm = MNISTPreprocessor.normalize(x_test)
        
        # 步驟 2: 攤平
        print("  [2/6] 攤平圖像 (28x28 → 784)...")
        x_train_flat = MNISTPreprocessor.flatten(x_train_norm)
        x_test_flat = MNISTPreprocessor.flatten(x_test_norm)
        
        # 步驟 3: 標準化
        print("  [3/6] 標準化特徵...")
        x_train_std, train_mean, train_std = MNISTPreprocessor.standardize(x_train_flat)
        x_test_std = (x_test_flat - train_mean) / (train_std + 1e-8)
        
        # 步驟 4: 數據增強
        print("  [4/6] 應用數據增強...")
        x_train_aug = MNISTPreprocessor.data_augmentation(x_train_std, intensity=0.05)
        
        # 步驟 5: One-hot 編碼
        print("  [5/6] 編碼標籤...")
        y_train_enc = MNISTPreprocessor.one_hot_encode(y_train)
        y_test_enc = MNISTPreprocessor.one_hot_encode(y_test)
        
        # 步驟 6: 統計資訊
        print("  [6/6] 生成統計資訊...")
        stats = {
            'train_shape': x_train_aug.shape,
            'test_shape': x_test_flat.shape,
            'train_mean': np.mean(x_train_aug),
            'train_std': np.std(x_train_aug),
            'test_mean': np.mean(x_test_std),
            'test_std': np.std(x_test_std),
            'train_min': np.min(x_train_aug),
            'train_max': np.max(x_train_aug),
        }
        
        print("✅ 前處理完成！\n")
        return x_train_aug, y_train_enc, x_test_std, y_test_enc, stats


# ===== 模型類別 =====
class MNISTNeuralNetwork:
    """MNIST 神經網路模型"""
    
    def __init__(self, n1=20, n2=20, n3=20, learning_rate=0.087):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """建立神經網路模型"""
        model = Sequential()
        model.add(Dense(self.n1, input_dim=784, activation='relu', name='hidden1'))
        model.add(Dense(self.n2, activation='relu', name='hidden2'))
        model.add(Dense(self.n3, activation='relu', name='hidden3'))
        model.add(Dense(10, activation='softmax', name='output'))
        
        model.compile(
            loss='mse',
            optimizer=SGD(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model
    
    def train(self, x_train, y_train, batch_size=100, epochs=10, validation_split=0.2):
        """訓練模型"""
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        return self.history
    
    def evaluate(self, x_test, y_test):
        """評估模型"""
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, acc
    
    def predict(self, x):
        """預測"""
        return self.model.predict(x, verbose=0)


# ===== 指標計算函式 =====
def calculate_metrics(y_true, y_pred, class_names=None):
    """計算詳細的分類指標"""
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    # 轉換為類別標籤
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # 計算各種指標
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    class_report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred_labels': y_pred_labels,
        'y_true_labels': y_true_labels
    }


# ===== Streamlit UI =====
def main():
    st.set_page_config(page_title="MNIST 手寫辨識系統", layout="wide", initial_sidebar_state="expanded")
    
    # 標題和說明
    st.title("🧠 MNIST 手寫數字辨識系統")
    st.markdown("**擴展版本** - 包含豐富的前處理、指標分析和互動式預測")
    
    # 側邊欄 - 參數設定
    st.sidebar.header("⚙️ 模型配置")
    n1 = st.sidebar.slider("隱藏層 1 神經元數", 10, 100, N1)
    n2 = st.sidebar.slider("隱藏層 2 神經元數", 10, 100, N2)
    n3 = st.sidebar.slider("隱藏層 3 神經元數", 10, 100, N3)
    lr = st.sidebar.slider("學習率", 0.001, 0.1, LEARNING_RATE, step=0.001)
    epochs = st.sidebar.slider("訓練週期", 5, 20, EPOCHS)
    
    st.sidebar.header("📊 檢視選項")
    show_preprocessing = st.sidebar.checkbox("顯示前處理詳情", value=True)
    show_training = st.sidebar.checkbox("顯示訓練過程", value=True)
    show_metrics = st.sidebar.checkbox("顯示詳細指標", value=True)
    show_prediction = st.sidebar.checkbox("啟用即時預測", value=True)
    
    # 主要標籤
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📥 數據加載", "🔧 前處理分析", "📈 訓練過程", "📊 模型評估", "🎨 即時預測"]
    )
    
    # ===== 標籤 1: 數據加載 =====
    with tab1:
        st.header("數據加載")
        if st.button("🚀 加載 MNIST 數據集", key="load_data"):
            with st.spinner("正在加載數據..."):
                (x_train, y_train), (x_test, y_test) = MNISTPreprocessor.load_data()
                st.session_state.x_train_raw = x_train
                st.session_state.y_train_raw = y_train
                st.session_state.x_test_raw = x_test
                st.session_state.y_test_raw = y_test
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("訓練樣本數", len(x_train))
                st.metric("圖像尺寸", "28 × 28")
            with col2:
                st.metric("測試樣本數", len(x_test))
                st.metric("類別數", 10)
        
        # 顯示樣本圖像
        if 'x_train_raw' in st.session_state:
            st.subheader("樣本圖像展示")
            sample_idx = st.slider("選擇樣本索引", 0, 59999, 0)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(st.session_state.x_train_raw[sample_idx], cmap='Greys')
                ax.set_title(f"訓練樣本 #{sample_idx}\n標籤: {st.session_state.y_train_raw[sample_idx]}", fontsize=14)
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.write("**像素值統計:**")
                img = st.session_state.x_train_raw[sample_idx]
                st.info(f"""
                最小值: {np.min(img)} | 最大值: {np.max(img)}
                平均值: {np.mean(img):.2f} | 標準差: {np.std(img):.2f}
                像素個數: {img.size}
                """)
    
    # ===== 標籤 2: 前處理分析 =====
    with tab2:
        st.header("前處理流程分析")
        if 'x_train_raw' not in st.session_state:
            st.warning("⚠️ 請先在'數據加載'標籤加載數據")
        else:
            if st.button("🔧 執行前處理", key="preprocess"):
                with st.spinner("前處理中..."):
                    x_train_processed, y_train_enc, x_test_processed, y_test_enc, stats = \
                        MNISTPreprocessor.preprocess_pipeline(
                            st.session_state.x_train_raw,
                            st.session_state.y_train_raw,
                            st.session_state.x_test_raw,
                            st.session_state.y_test_raw
                        )
                    st.session_state.x_train = x_train_processed
                    st.session_state.y_train = y_train_enc
                    st.session_state.x_test = x_test_processed
                    st.session_state.y_test = y_test_enc
                    st.session_state.stats = stats
                
                st.success("✅ 前處理完成！")
            
            if show_preprocessing and 'stats' in st.session_state:
                st.subheader("前處理統計結果")
                stats = st.session_state.stats
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("訓練集均值", f"{stats['train_mean']:.4f}")
                    st.metric("訓練集標差", f"{stats['train_std']:.4f}")
                with col2:
                    st.metric("測試集均值", f"{stats['test_mean']:.4f}")
                    st.metric("測試集標差", f"{stats['test_std']:.4f}")
                with col3:
                    st.metric("訓練集最小值", f"{stats['train_min']:.4f}")
                    st.metric("訓練集最大值", f"{stats['train_max']:.4f}")
                
                # 可視化前後對比
                if 'x_train_raw' in st.session_state:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    sample_idx = st.slider("對比樣本索引", 0, 59999, 0, key="compare_idx")
                    
                    # 原始圖像
                    axes[0].imshow(st.session_state.x_train_raw[sample_idx], cmap='Greys')
                    axes[0].set_title("原始圖像")
                    axes[0].axis('off')
                    
                    # 正規化後
                    axes[1].imshow(
                        MNISTPreprocessor.flatten(
                            MNISTPreprocessor.normalize(st.session_state.x_train_raw)
                        )[sample_idx].reshape(28, 28),
                        cmap='Greys'
                    )
                    axes[1].set_title("正規化後")
                    axes[1].axis('off')
                    
                    # 完全前處理後
                    axes[2].imshow(
                        st.session_state.x_train[sample_idx].reshape(28, 28),
                        cmap='Greys'
                    )
                    axes[2].set_title("前處理後")
                    axes[2].axis('off')
                    
                    st.pyplot(fig)
    
    # ===== 標籤 3: 訓練過程 =====
    with tab3:
        st.header("模型訓練")
        if 'x_train' not in st.session_state:
            st.warning("⚠️ 請先完成數據加載和前處理")
        else:
            if st.button("🚂 開始訓練", key="train"):
                with st.spinner("訓練中..."):
                    nn = MNISTNeuralNetwork(n1=n1, n2=n2, n3=n3, learning_rate=lr)
                    
                    st.write("模型架構:")
                    st.code(f"""
Layer 1: Dense(784) → ReLU → Dense({n1})
Layer 2: Dense({n1}) → ReLU → Dense({n2})
Layer 3: Dense({n2}) → ReLU → Dense({n3})
Layer 4: Dense({n3}) → Softmax → Dense(10)
                    """)
                    
                    history = nn.train(
                        st.session_state.x_train,
                        st.session_state.y_train,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=VALIDATION_SPLIT
                    )
                    
                    st.session_state.model = nn
                    st.session_state.history = history
                
                st.success("✅ 訓練完成！")
            
            if show_training and 'history' in st.session_state:
                history = st.session_state.history
                
                # 訓練曲線
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(history.history['loss'], label='訓練 Loss', linewidth=2)
                    ax.plot(history.history['val_loss'], label='驗證 Loss', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('損失函數變化')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(history.history['accuracy'], label='訓練精度', linewidth=2)
                    ax.plot(history.history['val_accuracy'], label='驗證精度', linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('準確率變化')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    # ===== 標籤 4: 模型評估 =====
    with tab4:
        st.header("模型評估")
        if 'model' not in st.session_state:
            st.warning("⚠️ 請先訓練模型")
        else:
            if st.button("📊 評估模型", key="evaluate"):
                with st.spinner("評估中..."):
                    loss, acc = st.session_state.model.evaluate(
                        st.session_state.x_test,
                        st.session_state.y_test
                    )
                    
                    y_pred = st.session_state.model.predict(st.session_state.x_test)
                    metrics = calculate_metrics(st.session_state.y_test, y_pred)
                    
                    st.session_state.metrics = metrics
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("測試損失", f"{loss:.4f}")
                with col2:
                    st.metric("測試精度", f"{acc*100:.2f}%")
            
            if show_metrics and 'metrics' in st.session_state:
                metrics = st.session_state.metrics
                
                # 混淆矩陣
                st.subheader("混淆矩陣")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': '樣本數'})
                ax.set_xlabel('預測標籤')
                ax.set_ylabel('真實標籤')
                st.pyplot(fig)
                
                # 分類報告
                st.subheader("分類報告 (逐類精度)")
                report_df = st.session_state.metrics['classification_report']
                
                # 轉換為 DataFrame
                report_data = []
                for digit in range(10):
                    digit_str = str(digit)
                    if digit_str in report_df:
                        report_data.append({
                            '數字': digit,
                            '精度': f"{report_df[digit_str]['precision']:.3f}",
                            '召回率': f"{report_df[digit_str]['recall']:.3f}",
                            'F1-分數': f"{report_df[digit_str]['f1-score']:.3f}",
                            '樣本數': int(report_df[digit_str]['support'])
                        })
                
                st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    # ===== 標籤 5: 即時預測 =====
    with tab5:
        st.header("即時手寫數字辨識")
        if 'model' not in st.session_state:
            st.warning("⚠️ 請先訓練模型")
        elif show_prediction:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("從測試集選擇")
                test_idx = st.slider("選擇測試樣本", 0, len(st.session_state.x_test)-1, 0)
                
                # 顯示圖像
                img_flat = st.session_state.x_test[test_idx]
                img_2d = img_flat.reshape(28, 28)
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_2d, cmap='Greys')
                ax.set_title(f"測試樣本 #{test_idx}")
                ax.axis('off')
                st.pyplot(fig)
                
                # 預測
                pred_prob = st.session_state.model.predict(img_flat.reshape(1, -1))[0]
                pred_class = np.argmax(pred_prob)
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.metric("預測結果", pred_class, delta=f"信心度: {pred_prob[pred_class]*100:.1f}%")
                with col_right:
                    st.metric("真實標籤", int(np.argmax(st.session_state.y_test[test_idx])))
            
            with col2:
                st.subheader("預測概率分布")
                
                pred_prob = st.session_state.model.predict(
                    st.session_state.x_test[test_idx].reshape(1, -1)
                )[0]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                digits = list(range(10))
                colors = ['green' if i == np.argmax(pred_prob) else 'skyblue' for i in range(10)]
                ax.bar(digits, pred_prob, color=colors)
                ax.set_xlabel('數字')
                ax.set_ylabel('預測概率')
                ax.set_title('各數字的預測概率')
                ax.set_xticks(digits)
                st.pyplot(fig)


# ===== 執行 =====
if __name__ == "__main__":
    import pandas as pd
    main()
