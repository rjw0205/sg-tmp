# Get the username and sanitize it by replacing dots with underscores
USERNAME := $(shell echo $(USER) | tr '.' '_')

# Define the Docker Compose project name
COMPOSE_PROJECT_NAME := $(USERNAME)-sg-dev

# Create and start the dev container named $(USER)-miro.
up:
	@COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT_NAME) docker-compose up --build -d
	@COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT_NAME) docker-compose exec dev bash install_incl.sh

# This runs the shell from the active dev container.
shell:
	@COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT_NAME) docker-compose exec dev zsh

# This stops the dev container
down:
	@COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT_NAME) docker-compose down