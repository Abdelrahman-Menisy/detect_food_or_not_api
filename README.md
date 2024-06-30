# Detect Food or Not API

This project provides an API to detect whether an image contains food or not. The model is trained and deployed using FastAPI, and the entire application is containerized using Docker. The Docker image is available on Docker Hub and deployed on Render. This API is utilized in a Flutter application to help chefs determine if the photos they upload for their menus contain food.

## Table of Contents
- [Overview](#overview)
- [Model Training](#model-training)
- [API Implementation](#api-implementation)
- [Dockerization](#dockerization)
- [Deployment](#deployment)
- [Usage in Flutter App](#usage-in-flutter-app)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Detect Food or Not API leverages a machine learning model to classify images as containing food or not. The API is built using FastAPI and is containerized for easy deployment. The Docker image is available on Docker Hub and can be deployed to various platforms, including Render.

## Model Training
The model is trained using the 5kfood dataset, which consists of 5,000 images of food and non-food items. Various machine learning techniques and models are evaluated to find the best performing one. The final model is saved and used in the API.

## API Implementation
The API is implemented using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python 3.11.5 based on standard Python type hints.

Key Endpoints:
- `POST /predict`: Accepts an image file and returns whether the image contains food or not.

## Dockerization
The application is containerized using Docker. The Dockerfile sets up the environment, installs dependencies, and runs the FastAPI server.

## Deployment
The Docker image is deployed on Render. Render is a unified platform to build and run all your apps and websites with free SSL, a global CDN, private networks, and auto deploys from Git.

## Usage in Flutter App
The API is used in a Flutter application available on GitHub. The Flutter app allows chefs to upload images and allow the flutter developers to determine if the photos they upload for their menus are food.

Flutter App Repository: [GitHub - Mohamed15Ghaly](https://github.com/Mohamed15Ghaly)

## Getting Started

### Pull the Docker Image
To get started with the API, you can pull the Docker image from Docker Hub:

```sh
docker pull abdelrahmanmenisy2002/detect_food_api
