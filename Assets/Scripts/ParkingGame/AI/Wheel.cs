using UnityEngine;

namespace ParkingGame.AI
{
    [System.Serializable]
    public struct Wheel
    {
        public WheelCollider collider;
        public Transform transform;
        public bool steering;
        public bool accelerating;
        public bool braking;
    }
}