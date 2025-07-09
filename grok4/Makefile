.PHONY: help install dev build start test lint clean docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make dev         - Start development servers"
	@echo "  make build       - Build for production"
	@echo "  make start       - Start production server"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linters"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docker-up   - Start Docker containers"
	@echo "  make docker-down - Stop Docker containers"

# Install dependencies
install:
	npm install
	cd backend && npm install
	cd frontend && npm install

# Development
dev:
	npm run dev

dev-backend:
	cd backend && npm run dev

dev-frontend:
	cd frontend && npm run dev

# Build
build:
	npm run build

build-backend:
	cd backend && npm run build

build-frontend:
	cd frontend && npm run build

# Production
start:
	cd backend && npm start

# Testing
test:
	npm test

test-backend:
	cd backend && npm test

test-frontend:
	cd frontend && npm test

# Linting
lint:
	npm run lint

lint-backend:
	cd backend && npm run lint

lint-frontend:
	cd frontend && npm run lint

# Clean
clean:
	rm -rf backend/dist
	rm -rf frontend/dist
	rm -rf backend/node_modules
	rm -rf frontend/node_modules
	rm -rf node_modules

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-rebuild:
	docker-compose build --no-cache

# Database
db-seed:
	cd backend && npm run seed

db-migrate:
	cd backend && npm run migrate

# Environment
setup-env:
	cp backend/.env.example backend/.env
	@echo "Environment file created. Please edit backend/.env with your configuration."