using System;
using ParkingGame.UI;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

namespace ParkingGame.AI
{
    public class CarAgentWithContinuousController: Agent
    {
        private Rigidbody _carRb;
        private float _lastDistance;

        // Fields for the actions
        private float _accelerateInput;
        private float _turnInput;
        private float _brakeInput;

        // Fields for the target and spawn location
        private Transform _target;
        private Transform _spawnArea;
        private GameObject[] _environments;
        private GameObject _currentEnv;

        [Header("Driving parameters")] 
        [SerializeField] private float motorForce = 2000;
        [SerializeField] private float breakForce = 4000;
        [SerializeField] private float maxSteerAngle = 30;
        [SerializeField] private Transform centerOfMass;

        [Header("Wheels")] 
        [SerializeField] private Wheel flWheel;
        [SerializeField] private Wheel frWheel;
        [SerializeField] private Wheel rlWheel;
        [SerializeField] private Wheel rrWheel;

        [Header("Heuristic")] 
        [SerializeField] private string accelerateButtonName = "Vertical";
        [SerializeField] private string turnInputName = "Horizontal";
        [SerializeField] private KeyCode brakeButtonName = KeyCode.Space;

        [Header("Environments")] 
        [SerializeField] private string pathToEnvironmentPrefabs;
        
        public override void Initialize()
        {
            _carRb = gameObject.GetComponent<Rigidbody>();
            _environments = Resources.LoadAll<GameObject>(pathToEnvironmentPrefabs);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 relativePosition = transform.position - _target.position;
            sensor.AddObservation(relativePosition);
            sensor.AddObservation(_carRb.velocity.magnitude);
            sensor.AddObservation(transform.forward);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            // apply our physics properties
            _carRb.centerOfMass = transform.InverseTransformPoint(centerOfMass.position);

            // Take the action
            InterpretActions(actionBuffers);
            HandleAcceleration();
            HandleBreaking();
            HandleSteering();
            UpdateWheels();

            // Add reward depending on whether the agent moves in the direction of the parking lot
            float newDistance = Vector3.Distance(_target.position, transform.position);
            float reward = _lastDistance - newDistance;
            AddReward(reward);
            _lastDistance = newDistance;

            // Add a negative reward for the time step
            AddReward(-0.1f);

            // End episode
            if (Vector3.Distance(_target.position, transform.position) < 1.5f && _carRb.velocity.sqrMagnitude <= 0.2f)
            {
                AddReward(300);
                EndEpisode();
            }
        }

        public override void OnEpisodeBegin()
        {
            // Remove the old environment
            if (_currentEnv)
            {
                Destroy(_currentEnv);
                _currentEnv = null;
            }

            // Instantiate a new environment
            int index = Random.Range(0, _environments.Length);
            _currentEnv = Instantiate(_environments[index], Vector3.zero, Quaternion.identity);

            // Get Target and Spawn Location from environment
            _target = _currentEnv.transform.Find("Parkplatz_Target");
            _spawnArea = _currentEnv.transform.Find("SpawnLocation");
            _lastDistance = Vector3.Distance(_target.position, transform.position);

            // find a random position for the agent to start from
            transform.position = ChooseRandomPosition();
            _carRb.velocity = Vector3.zero;
            _carRb.angularVelocity = Vector3.zero;

            // random orientation
            transform.Rotate(Vector3.up, Random.Range(0f, 360f));
        }

        public void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("Obstacle"))
            {
                AddReward(-10f);
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActionsOut = actionsOut.ContinuousActions;
            continuousActionsOut[0] = Input.GetAxis(accelerateButtonName);
            continuousActionsOut[1] = Input.GetAxis(turnInputName);
            continuousActionsOut[2] = Input.GetKey(brakeButtonName) ? 1f : 0f;
        }

        private void InterpretActions(in ActionBuffers actionBuffers)
        {
            var continuousActions = actionBuffers.ContinuousActions;
            _accelerateInput = Mathf.Clamp(continuousActions[0], -1f, 1f);
            _turnInput = Mathf.Clamp(continuousActions[1], -1f, 1f);
            _brakeInput = Mathf.Clamp(continuousActions[2], 0f, 1f);
            
            // update UI
            ActionsTextBehaviour.UpdateActions(_accelerateInput, _turnInput, _brakeInput);
        }

        private void HandleAcceleration()
        {
            // decrease the speed for moving backwards
            var scale = _accelerateInput > 0 ? 1f : 0.5f;
            var currentAccelerationForce = scale * _accelerateInput * motorForce;
            flWheel.collider.motorTorque = flWheel.accelerating ? currentAccelerationForce : 0;
            frWheel.collider.motorTorque = frWheel.accelerating ? currentAccelerationForce : 0;
            rlWheel.collider.motorTorque = rlWheel.accelerating ? currentAccelerationForce : 0;
            rrWheel.collider.motorTorque = rrWheel.accelerating ? currentAccelerationForce : 0;
        }

        private void HandleBreaking()
        {
            var currentBreakForce = _brakeInput * breakForce;
            flWheel.collider.brakeTorque = flWheel.braking ? currentBreakForce : 0;
            frWheel.collider.brakeTorque = frWheel.braking ? currentBreakForce : 0;
            rlWheel.collider.brakeTorque = rlWheel.braking ? currentBreakForce : 0;
            rrWheel.collider.brakeTorque = rrWheel.braking ? currentBreakForce : 0;
        }

        private void HandleSteering()
        {
            var currentSteerAngle = maxSteerAngle * _turnInput;
            flWheel.collider.steerAngle = flWheel.steering ? currentSteerAngle : 0;
            frWheel.collider.steerAngle = frWheel.steering ? currentSteerAngle : 0;
            rlWheel.collider.steerAngle = rlWheel.steering ? currentSteerAngle : 0;
            rrWheel.collider.steerAngle = rrWheel.steering ? currentSteerAngle : 0;
        }

        private void UpdateWheels()
        {
            UpdateSingleWheel(flWheel);
            UpdateSingleWheel(frWheel);
            UpdateSingleWheel(rlWheel);
            UpdateSingleWheel(rrWheel);
        }

        private void UpdateSingleWheel(Wheel wheel)
        {
            wheel.collider.GetWorldPose(out var pos, out var rot);
            wheel.transform.rotation = rot;
            wheel.transform.position = pos;
        }

        private Vector3 ChooseRandomPosition()
        {
            var size = _spawnArea.transform.localScale - new Vector3(1, 0, 1);
            var center = _spawnArea.transform.position;

            return center + new Vector3((Random.value - 0.5f) * size.x, 0, (Random.value - 0.5f) * size.z);
        }
    }
}