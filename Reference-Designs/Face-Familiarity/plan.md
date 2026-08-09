I want to build a prototype for scoring how familiar a face is to our dataset. 

The prototype takes in a video https://www.youtube.com/watch?v=r8Ch2JrDCZU

The prototype uses EyePop.ai in python to scan through for Person -> Face -> Face Embeddings to create a thumbnail library of people found, faces found, and embeddings. The pop definition is here:
Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    ability='eyepop.person.face.short-range:latest',
                    categoryName="2d-face-points",
                    forward=CropForward(
                        boxPadding=1.5,
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            ability='eyepop.face-id.base:latest'
                        )]
                    )
                )]
            )
        )
    ])

example output from EyePop.ai:
{
  "objects": [
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.9699,
      "height": 550.408,
      "objects": [
        {
          "classLabel": "face",
          "confidence": 0.5349,
          "height": 80.586,
          "raw": [
            {
              "confidence": 1,
              "tensors": [
                {
                  "data": [
                    [
                      0.0013,
                      -0.0396,
                      0.0666,
                      ...
                    ]
                  ],
                  "dimensions": [
                    512,
                    1
                  ],
                  "name": "embedding",
                  "type": "float32"
                }
              ]
            }
          ],
          "width": 80.586,
          "x": 308.964,
          "y": 128.766
        }
      ],
      "orientation": 0,
      "width": 165.56,
      "x": 264.49,
      "y": 89.591
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.9473,
      "height": 465.914,
      "orientation": 0,
      "width": 120.812,
      "x": 174.146,
      "y": 167.866
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.9206,
      "height": 63.648,
      "orientation": 0,
      "width": 28.929,
      "x": 294.685,
      "y": 202.832
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.9182,
      "height": 110.225,
      "orientation": 0,
      "width": 38.851,
      "x": 68.75,
      "y": 182.641
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.9155,
      "height": 381.46,
      "orientation": 0,
      "width": 40.204,
      "x": 0,
      "y": 213.629
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.874,
      "height": 285.182,
      "orientation": 0,
      "width": 73.819,
      "x": 4.693,
      "y": 196.95
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.8703,
      "height": 59.268,
      "orientation": 0,
      "width": 28.566,
      "x": 168.83,
      "y": 198.861
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.833,
      "height": 21.088,
      "orientation": 0,
      "width": 15.839,
      "x": 105.337,
      "y": 188.94
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.8294,
      "height": 260.214,
      "orientation": 0,
      "width": 45.897,
      "x": 434.102,
      "y": 249.976
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.8272,
      "height": 41.378,
      "orientation": 0,
      "width": 17.065,
      "x": 60.677,
      "y": 183.828
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.8267,
      "height": 39.924,
      "orientation": 0,
      "width": 16.443,
      "x": 277.277,
      "y": 216.973
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.8015,
      "height": 26.683,
      "orientation": 0,
      "width": 16.125,
      "x": 319.692,
      "y": 200.917
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.7998,
      "height": 80.25,
      "orientation": 0,
      "width": 23.666,
      "x": 429.669,
      "y": 211.63
    },
    {
      "category": "person",
      "classLabel": "person",
      "confidence": 0.7905,
      "height": 73.897,
      "objects": [
        {
          "classLabel": "face",
          "confidence": 0.812,
          "height": 14.421,
          "raw": [
            {
              "confidence": 1,
              "tensors": [
                {
                  "data": [
                    [
                      0.0013,
                      -0.0396,
                      0.0666,
                      ...
                    ]
                  ],
                  "dimensions": [
                    512,
                    1
                  ],
                  "name": "embedding",
                  "type": "float32"
                }
              ]
            }
          ],
          "width": 14.421,
          "x": 463.074,
          "y": 220.011
        }
      ],
      "orientation": 0,
      "width": 36.496,
      "x": 443.503,
      "y": 215.52
    }
  ],
  "seconds": 0,
  "source_height": 640,
  "source_id": "cdc93b2f-91f8-11f1-8787-a2332d7e776d",
  "source_width": 480,
  "system_timestamp": 1786063322929505000,
  "timestamp": 0
}

The embeddings are stored in a local db for vector search

There is an interface (html) to group and label face embeddings

There is an interface (html) to take in new data from a video: http://youtube.com/watch?v=IPlxtFnTXYI and label it with a Name from our database or Unknown becasue it's too dissimiliar

