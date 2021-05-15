using System;
using System.Collections;
using System.Collections.Generic;
using ParkingGame.CarSystem.Inputs;
using UnityEngine;

namespace ParkingGame.CarSystem
{
    
    public class CarController : MonoBehaviour
    {
        public InputData Input{ get; private set; }
        
        // the input sources that can control the car
        private IInput[] _inputs;
        private Rigidbody _car_rb;
        
        [SerializeField] private float motorForce = 800;
        [SerializeField] private float breakForce = 1600;
        [SerializeField] private float maxSteerAngle = 30;

        [SerializeField] private Transform centerOfMass;
        [SerializeField] private WheelCollider frontLeftWheelCollider;
        [SerializeField] private WheelCollider frontRightWheelCollider;
        [SerializeField] private WheelCollider rearLeftWheelCollider;
        [SerializeField] private WheelCollider rearRightWheelCollider;

        [SerializeField] private Transform frontLeftWheelTransform;
        [SerializeField] private Transform frontRightWheeTransform;
        [SerializeField] private Transform rearLeftWheelTransform;
        [SerializeField] private Transform rearRightWheelTransform;

        private void Awake()
        {
            _inputs = GetComponents<IInput>();
            _car_rb = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            // apply our physics properties
            _car_rb.centerOfMass = transform.InverseTransformPoint(centerOfMass.position);
            
            GatherInputs();
            HandleAcceleration();
            HandleBreaking();
            HandleSteering();
            UpdateWheels();
        }

        private void GatherInputs()
        {
            // reset input
            Input = new InputData();
            
            // gather nonzero inputs from our sources
            foreach (var localInput in _inputs)
            {
                Input = localInput.GenerateInput();
            } 
        }

        private void HandleAcceleration()
        {
            var currentAccelerationForce = Input.AccelerateInput * motorForce;
            frontLeftWheelCollider.motorTorque =  currentAccelerationForce;
            frontRightWheelCollider.motorTorque = currentAccelerationForce;
        }

        private void HandleBreaking()
        {
            var currentBreakForce = Input.Brake ? breakForce : 0f;
            frontLeftWheelCollider.brakeTorque = currentBreakForce;
            frontRightWheelCollider.brakeTorque = currentBreakForce;
            rearLeftWheelCollider.brakeTorque = currentBreakForce;
            rearRightWheelCollider.brakeTorque = currentBreakForce;
        }

        private void HandleSteering()
        {
            var currentSteerAngle = maxSteerAngle * Input.TurnInput;
            frontLeftWheelCollider.steerAngle = currentSteerAngle;
            frontRightWheelCollider.steerAngle = currentSteerAngle;
        }
        
        private void UpdateWheels()
        {
            UpdateSingleWheel(frontLeftWheelCollider, frontLeftWheelTransform);
            UpdateSingleWheel(frontRightWheelCollider, frontRightWheeTransform);
            UpdateSingleWheel(rearLeftWheelCollider, rearLeftWheelTransform);
            UpdateSingleWheel(rearRightWheelCollider, rearRightWheelTransform);
        }

        private void UpdateSingleWheel(WheelCollider wheelCollider, Transform wheelTransform)
        {
            wheelCollider.GetWorldPose(out var pos, out var rot);
            wheelTransform.rotation = rot;
            wheelTransform.position = pos;
        }
    }
}

