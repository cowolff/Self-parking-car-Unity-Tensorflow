using System;
using System.Collections;
using System.Collections.Generic;
using ParkingGame.CarSystem.Inputs;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace ParkingGame.AI
{
    public class CarAgent : Agent, IInput
    {
        private Rigidbody _car_rb;
        private float _accelerateInput;
        private float _turnInput;
        private bool _brake;
        private GameObject _target;
        private float _last_distance;
        private GameObject _spawnArea;
        
        public override void Initialize()
        {
            base.Initialize();
            _car_rb = gameObject.GetComponent<Rigidbody>();
            _target = GameObject.Find("Parkplatz_6");
            _spawnArea = GameObject.Find("SpawnLocation");

            _last_distance = Vector3.Distance(_target.transform.position, transform.position);
        }
    
        public override void CollectObservations(VectorSensor sensor)
        {

            Vector3 relativePosition = transform.position - _target.transform.position;
            float speed = _car_rb.velocity.magnitude;
            Vector3 orientation = transform.forward;

            // TODO: Add observations
            sensor.AddObservation(relativePosition);
            sensor.AddObservation(speed);
            sensor.AddObservation(orientation);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            InterpretActions(actionBuffers);

            float reward = _last_distance - Vector3.Distance(_target.transform.position, transform.position);

            // TODO: Negative reward when colliding with different objects

            AddReward(reward);

            if (Vector3.Distance(_target.transform.position, transform.position) < 3) {
                EndEpisode();
            }
        }

        public static Vector3 ChooseRandomPosition(Vector3 center, float minAngle, float maxAngle, float minRadius, float maxRadius)
        {
            float radius = minRadius;
            float angle = minAngle;

            if (maxRadius > minRadius)
            {
                // Pick a random radius
                radius = UnityEngine.Random.Range(minRadius, maxRadius);
            }

            if (maxAngle > minAngle)
            {
                // Pick a random angle
                angle = UnityEngine.Random.Range(minAngle, maxAngle);
            }

            // Center position + forward vector rotated around the Y axis by "angle" degrees, multiplies by "radius"
            return center + Quaternion.Euler(0f, angle, 0f) * Vector3.forward * radius;
        }

        public override void OnEpisodeBegin()
        {
            transform.position = ChooseRandomPosition(_spawnArea.transform.position, 0f, 360f, 0f, 3f) + Vector3.up * .5f;
        }
        
        private void InterpretActions(in ActionBuffers actionBuffers)
        {
            var continuousActions = actionBuffers.ContinuousActions;
            var discreteActions = actionBuffers.DiscreteActions;
            _accelerateInput = Mathf.Clamp(continuousActions[0], -1f, 1f);
            _turnInput = Mathf.Clamp(continuousActions[1], -1f, 1f);
            _brake = discreteActions[0] == 1;
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

