# Deep-Reinforcement-Learning-Self-driving-cars-in-unity

Our training loop saves the current model every 50 epochs so that in case of an error the current progress isn't lost. It is saved into the *saved_models* folder. Furthermore the current overall reward per training epoch is saved in a csv-file. The name of this file consists out of *log* and the unix time at the moment when the training is started.
