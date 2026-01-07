Privacy-Preserving Federated Learning (PPFL) with Differential Privacy

What’s This All About?

This project lets you train a machine learning model with a bunch of different clients, and nobody ever has to hand over their raw data. Each client does its own training locally. When it’s time to collaborate, they just send model updates—not the actual data—to a central server. The secret sauce is Differential Privacy, which keeps everyone’s personal info safe and sound.

What’s Cool About It

- Federated Learning: You get decentralized training. Multiple clients, one shared model, no data swapping.
- Differential Privacy: It adds noise to gradients, so nobody can peek at your sensitive details.
- Opacus Integration: Uses the DP-SGD optimizer for training that respects your privacy.
- MNIST Example: Shows how it all works with a well-known dataset.

Where Does This Actually Help?

- Healthcare: Hospitals can build smarter models using patient data, but they never send the records anywhere.
- Finance: Banks work together to spot fraud, but your transactions stay private.
- Mobile Devices: Your phone can help train text prediction or recommendation models, all without sending your texts to the cloud.
- IoT Devices: Gadgets like smart sensors detect weird behavior on their own, with no raw sensor data leaving the device.

How To Use It

1. Install what you need:

pip install torch torchvision opacus

2. Start training:

python main.py

You’ll see the training loss for each client and the total privacy budget (ε). Want to tweak things? Change the number of clients, local training epochs, learning rate, DP noise level, or gradient clipping—whatever fits your needs.

Why Bother?

Because it keeps your sensitive data protected while you still get the benefits of training with others. It helps you meet tough privacy rules like GDPR (General Data Protection Regulation) and HIPAA (Health Insurance Portability and Accountability Act). Basically, you can learn from a bigger pool of data without giving up your secrets.
