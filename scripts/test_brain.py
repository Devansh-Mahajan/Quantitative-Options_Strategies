import sys
import os

# Ensure Python can find the core directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.regime_detection import get_brain_prediction

def main():
    print("\n[🔌] Booting up the RTX 5090 Inference Engine...")
    print("[📥] Downloading last 60 days of macro data...")
    print("[🧠] Feeding sequence to Deep Mixture of Experts Model...")
    
    strategy, confidence, probs = get_brain_prediction()
    
    print("\n==================================================")
    print(f"🔥 FINAL VERDICT:  {strategy}")
    print(f"🎯 CONFIDENCE:     {confidence:.2f}%")
    print("==================================================")
    print("🧠 UNDER THE HOOD (Raw Probabilities):")
    print(f"   🟢 Theta Engine (Sell Premium):  {probs[0]*100:.2f}%")
    print(f"   🟡 Vega Sniper  (Volatility):    {probs[1]*100:.2f}%")
    print(f"   🔴 Tail Hedge   (Crash Guard):   {probs[2]*100:.2f}%")
    print("==================================================\n")

if __name__ == "__main__":
    main()