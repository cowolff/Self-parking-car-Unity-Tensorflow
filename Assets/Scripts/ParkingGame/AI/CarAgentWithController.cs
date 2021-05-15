using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

namespace ParkingGame.AI
{
    public class CarAgentWithController: Agent
    {
        private Rigidbody _carRb;
        private float _lastDistance;

        // Fields for the actions
        private float _accelerateInput;
        private float _turnInput;
        private float _brakeInput;
        
        // Fields for the target and spawn location
        [SerializeField] private GameObject target;
        [SerializeField] private GameObject spawnArea;
        
        // Driving related fields
        [SerializeField] private float motorForce = 9000;
        [SerializeField] private float breakForce = 18000;
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
        
        // Fields for the heuristic handling
        [SerializeField] private string accelerateButtonName = "Vertical";
        [SerializeField] private string turnInputName = "Horizontal";
        [SerializeField] private KeyCode brakeButtonName = KeyCode.Space;
        
        public override void Initialize()
        {
            _carRb = gameObject.GetComponent<Rigidbody>();
            
            _lastDistance = Vector3.Distance(target.transform.position, transform.position);
        }
        
        public override void CollectObservations(VectorSensor sensor)
        {
            System.Diagnostics.Debug.WriteLine("CollectObservations()");
            Vector3 relativePosition = transform.position - target.transform.position;
            sensor.AddObservation(relativePosition);
            sensor.AddObservation(_carRb.velocity.magnitude);
            sensor.AddObservation(transform.forward);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            System.Diagnostics.Debug.WriteLine("OnActionReceived()");
            // apply our physics properties
            _carRb.centerOfMass = transform.InverseTransformPoint(centerOfMass.position);
            
            // Take the action
            InterpretActions(actionBuffers);
            HandleAcceleration();
            HandleBreaking();
            HandleSteering();
            UpdateWheels();
            
            // Add reward depending on whether the agent moves in the direction of the parking lot
            float newDistance = Vector3.Distance(target.transform.position, transform.position);
            float reward = _lastDistance - newDistance;
            AddReward(reward);
            _lastDistance = newDistance;
            
            // Add a negative reward for the time step
            AddReward(-0.1f);

            // End episode
            if (Vector3.Distance(target.transform.position, transform.position) < 1.5f && _carRb.velocity.sqrMagnitude <= 0.2f) {
                AddReward(300);
                EndEpisode();
            }
        }
        
        public override void OnEpisodeBegin()
        {
            // find a random position for the agent to start from
            System.Diagnostics.Debug.WriteLine("OnEpisode()");
            transform.position = ChooseRandomPosition();
            _carRb.velocity = Vector3.zero;
            _carRb.angularVelocity = Vector3.zero;
            
            // random orientation
            transform.Rotate(Vector3.up, Random.Range(0f, 360f));
        }
        
        public void OnCollisionEnter(Collision collision)
        {
            if(collision.gameObject.tag == "Obstacle")
            {
                AddReward(-20f);
            }
        }
        
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActionsOut = actionsOut.DiscreteActions;
            discreteActionsOut[0] = 0;
            discreteActionsOut[1] = 0;
            discreteActionsOut[2] = Input.GetKey(brakeButtonName) ? 1 : 0;
            
            if (Input.GetAxis(accelerateButtonName) > 0.1f)
            {
                discreteActionsOut[0] = 4;
            }
            else if (Input.GetAxis(accelerateButtonName) < -0.1f)
            {
                discreteActionsOut[0] = 6;
            }
            
            if (Input.GetAxis(turnInputName) > 0.1f)
            {
                discreteActionsOut[1] = 3;
            }
            else if (Input.GetAxis(turnInputName) < -0.1f)
            {
                discreteActionsOut[1] = 6;
            }
        }
        
        private void InterpretActions(in ActionBuffers actionBuffers)
        {
            var discreteActions = actionBuffers.DiscreteActions;
            switch (discreteActions[0])
            {
                case 0:
                    _accelerateInput = 0f;
                    break;
                case 1:
                    _accelerateInput = 0.25f;
                    break;
                case 2:
                    _accelerateInput = 0.5f;
                    break;
                case 3:
                    _accelerateInput = 0.75f;
                    break;
                case 4:
                    _accelerateInput = 1f;
                    break;
                case 5:
                    _accelerateInput = -0.2f;
                    break;
                case 6:
                    _accelerateInput = -0.4f;
                    break;
            }
            
            switch (discreteActions[1])
            {
                case 0:
                    _turnInput = 0f;
                    break;
                case 1:
                    _turnInput = 0.33f;
                    break;
                case 2:
                    _turnInput = 0.67f;
                    break;
                case 3:
                    _turnInput = 1f;
                    break;
                case 4:
                    _turnInput = -0.33f;
                    break;
                case 5:
                    _turnInput = -0.67f;
                    break;
                case 6:
                    _turnInput = -1f;
                    break;
            }

            _brakeInput = discreteActions[2];
        }
        
        private void HandleAcceleration()
        {
            var currentAccelerationForce = _accelerateInput * motorForce;
            frontLeftWheelCollider.motorTorque =  currentAccelerationForce;
            frontRightWheelCollider.motorTorque = currentAccelerationForce;
        }

        private void HandleBreaking()
        {
            var currentBreakForce = _brakeInput * breakForce;
            frontLeftWheelCollider.brakeTorque = currentBreakForce;
            frontRightWheelCollider.brakeTorque = currentBreakForce;
            rearLeftWheelCollider.brakeTorque = currentBreakForce;
            rearRightWheelCollider.brakeTorque = currentBreakForce;
        }

        private void HandleSteering()
        {
            var currentSteerAngle = maxSteerAngle * _turnInput;
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
        
        private Vector3 ChooseRandomPosition()
        {
            var size = spawnArea.transform.localScale - new Vector3(1, 0, 1);
            var center = spawnArea.transform.position;
            
            return center + new Vector3((Random.value - 0.5f) * size.x, 0, (Random.value - 0.5f) * size.z);
        }
    }
}