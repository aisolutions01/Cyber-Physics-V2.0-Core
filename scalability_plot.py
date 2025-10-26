import matplotlib.pyplot as plt
import numpy as np

# بيانات من نتائجك الفعلية (ملخص من model_example.py)
batches = np.arange(1, 11)
accuracy = [0.480, 0.536, 0.514, 0.550, 0.529, 0.486, 0.529, 0.571, 0.486, 0.514]
latency = [0.04, 0.05, 0.06, 0.06, 0.05, 0.03, 0.04, 0.05, 0.04, 0.03]  # بالثواني التقريبية

# إنشاء الشكل والمخطط المزدوج المحور
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
ax1.set_xlabel('Batch Number (100 incidents each)')
ax1.set_ylabel('Accuracy', color=color1)
ax1.plot(batches, accuracy, marker='o', color=color1, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.4, 0.6)
ax1.grid(True, linestyle='--', alpha=0.5)

# المحور الثاني لزمن التدريب (latency)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Training Latency (seconds)', color=color2)
ax2.plot(batches, latency, marker='s', linestyle='--', color=color2, label='Latency')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 0.08)

# العنوان والتوضيح
plt.title('Scalability Analysis — On-the-Fly Incremental Learning')
fig.tight_layout()

# حفظ الصورة
plt.savefig('scalability.png', dpi=300)
plt.show()
