using System;
using System.Collections;
using System.Collections.Generic;
using ParkingGame.CarSystem.Inputs;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

namespace ParkingGame.AI
{
    public class CarAgent : Agent, IInput
    {
        private Rigidbody _carRb;
        private float _accelerateInput;
        private float _turnInput;
        private bool _brake;
        private GameObject _target;
        private float _lastDistance;
        private GameObject _spawnArea;
        
        public override void Initialize()
        {
            // base.Initialize();
            _carRb = gameObject.GetComponent<Rigidbody>();
            _target = GameObject.Find("Parkplatz_6");
            _spawnArea = GameObject.Find("SpawnLocation");

            _lastDistance = Vector3.Distance(_target.transform.position, transform.position);
        }
    
        public override void CollectObservations(VectorSensor sensor)
        {

            Vector3 relativePosition = transform.position - _target.transform.position;
            float speed = _carRb.velocity.magnitude;
            Vector3 orientation = transform.forward;

            sensor.AddObservation(relativePosition);
            sensor.AddObservation(speed);
            sensor.AddObservation(orientation);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            InterpretActions(actionBuffers);

            // Add reward depending on whether the agent moves in the direction to the parking lot
            float reward = _lastDistance - Vector3.Distance(_target.transform.position, transform.position);
            AddReward(reward);

            // End episode
            if (Vector3.Distance(_target.transform.position, transform.position) < 3) {
                AddReward(100);
                EndEpisode();
            }
        }

        private Vector3 ChooseRandomPosition()
        {
            var size = _spawnArea.transform.localScale - new Vector3(1, 0, 1);
            var center = _spawnArea.transform.position;
            
            return center + new Vector3((Random.value - 0.5f) * size.x, 0, (Random.value - 0.5f) * size.z);
        }

        public override void OnEpisodeBegin()
        {
            // find a random position for the agent to start from
            _carRb.velocity = Vector3.zero;
            _carRb.angularVelocity = Vector3.zero;
            transform.position = ChooseRandomPosition();
            
        }
        
        private void InterpretActions(in ActionBuffers actionBuffers)
        {
            var continuousActions = actionBuffers.ContinuousActions;
            //var discreteActions = actionBuffers.DiscreteActions;
            _accelerateInput = Mathf.Clamp(continuousActions[0], -1f, 1f);
            _turnInput = Mathf.Clamp(continuousActions[1], -1f, 1f);
            _brake = false; //discreteActions[0] == 1;
        }

        public InputData GenerateInput()
        {
            return new InputData
            {
                AccelerateInput = _accelerateInput,
                TurnInput = _turnInput,
                Brake = _brake,
            };
        }

        public void onCollisionEnter(Collision collision)
        {
            AddReward(-10);
        }
    }
}

