#!/bin/bash
gunicorn -w 1 -b 0.0.0.0:5000 food_quality_api:app
