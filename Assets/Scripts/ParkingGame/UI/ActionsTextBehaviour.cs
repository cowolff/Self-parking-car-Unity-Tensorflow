using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ParkingGame.UI
{
    public class ActionsTextBehaviour : MonoBehaviour
    {
        private Text _actionInputText;
        private const string ActionInputTextTemplate = "A: {0}\nS: {1}\nB: {2}";

        private static float Acceleration, Steering, Braking = 0f;
        
        public void Start()
        {
            _actionInputText = GetComponent<Text>();
        }

        public void Update()
        {
            _actionInputText.text = String.Format(ActionInputTextTemplate, Acceleration, Steering, Braking);
        }

        public static void UpdateActions(float acceleration, float steering, float braking)
        {
            Acceleration = acceleration;
            Steering = steering;
            Braking = braking;
        }
    }
}