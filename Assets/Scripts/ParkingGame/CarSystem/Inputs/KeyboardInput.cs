using UnityEngine;

namespace ParkingGame.CarSystem.Inputs
{
    public class KeyboardInput : MonoBehaviour, IInput
    {
        [SerializeField] private string accelerateButtonName = "Vertical";
        [SerializeField] private string turnInputName = "Horizontal";
        [SerializeField] private KeyCode brakeButtonName = KeyCode.Space;
        
        public InputData GenerateInput()
        {
            return new InputData
            {
                AccelerateInput = Input.GetAxis(accelerateButtonName),
                TurnInput = Input.GetAxis(turnInputName),
                Brake = Input.GetKey(brakeButtonName)
            };
        }
    }
}