# 🐳 Docker Revision Notes

> Simplified & ready-for-interview Docker notes

---

## 📦 What is Docker?

Docker is a platform to **build, ship, and run** applications using containers.

- Containers are lightweight, portable, and run isolated from each other.
- Docker uses OS-level virtualization (not VMs).

---

## 🧰 Basic Docker Commands

### 🔹 Images

```bash
docker pull <image-name>                # Download image from Docker Hub
docker images                           # List all images
docker rmi <image-id>                   # Remove image
```

### 🔹 Containers

```bash
docker run <image-name>                # Run a container
docker run -it <image> bash            # Run with interactive shell
docker run -d <image>                  # Run in detached mode
docker run -p 8080:80 <image>          # Port binding: host:container

docker ps                              # List running containers
docker ps -a                           # List all containers
docker stop <container-id>            # Stop container
docker start <container-id>           # Start container
docker restart <container-id>         # Restart container
docker rm <container-id>              # Remove container
```

### 🔹 Exec & Logs

```bash
docker exec -it <container-id> bash   # Access running container shell
docker logs <container-id>           # View container logs
```

### 🔹 Volumes (Data Persistence)

```bash
docker volume create <volume-name>               # Create volume
docker volume ls                                 # List volumes
docker run -v <volume-name>:/path/in/container   # Mount volume
```

---

## ⚙️ Dockerfile

Used to define steps to build a custom image.

### 🔸 Sample Dockerfile

```Dockerfile
FROM node:16
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "index.js"]
```

### 🔸 Build & Run from Dockerfile

```bash
docker build -t myapp-image .
docker run -p 3000:3000 myapp-image
```

---

## 📁 Docker Compose (Multi-container apps)

### 🔸 Sample `docker-compose.yml`

```yaml
version: '3'
services:
  web:
    image: nginx
    ports:
      - "8080:80"
  db:
    image: mysql
    environment:
      MYSQL_ROOT_PASSWORD: example
```

### 🔸 Commands

```bash
docker-compose up                      # Start services
docker-compose up -d                   # Detached mode
docker-compose down                    # Stop and remove containers
```

---

## 🧠 Important Concepts

| Concept          | Description |
|------------------|-------------|
| **Image**        | Read-only template used to create containers |
| **Container**    | Running instance of an image |
| **Volume**       | Used for persistent data |
| **Network**      | Custom bridge for container communication |
| **Dockerfile**   | Blueprint to create custom images |
| **Docker Hub**   | Public registry for Docker images |
| **Detached Mode**| Runs container in background (`-d`) |
| **Port Mapping** | Connect container ports to host (`-p 8080:80`) |

---

## 💡 Tips

- Always `.dockerignore` files like `node_modules/`, `*.log`, etc.
- Use named volumes for data persistence across container restarts.
- Prefer multi-stage builds for smaller images.
- Clean up regularly:
  ```bash
  docker system prune
  docker image prune
  ```

---

## 📚 To Add Later

- Docker Networking (bridge, host, overlay)
- Docker Swarm / Kubernetes basics
- Security Best Practices
- Multi-stage Dockerfiles
- Caching & Layering optimizations
- Private Docker Registries
---

## 🌐 Docker Networking

### 🔸 Types of Networks

| Network Type | Description |
|--------------|-------------|
| **Bridge**   | Default network; containers can communicate using IP |
| **Host**     | Shares host's networking stack; no isolation |
| **Overlay**  | Used in Docker Swarm; allows containers on different hosts to communicate |

```bash
docker network ls                            # List networks
docker network create <name>                 # Create custom bridge network
docker network inspect <network-name>        # View network details
docker network connect <network> <container> # Connect container to network
```

---

## 🐝 Docker Swarm & ☸️ Kubernetes (Basics)

### 🔹 Docker Swarm

- Native clustering and orchestration tool for Docker
- Use `docker swarm init` to create a Swarm
- Define services using `docker service` commands

```bash
docker swarm init
docker service create --replicas 3 -p 8080:80 nginx
docker service ls
```

### 🔹 Kubernetes

- Powerful container orchestration system for scaling, management
- Uses YAML for deployment files
- Components: Pod, Deployment, Service, Ingress

```bash
kubectl get pods
kubectl apply -f deployment.yaml
kubectl delete -f deployment.yaml
```

---

## 🔐 Docker Security Best Practices

- Use minimal base images (e.g., alpine)
- Do not run as root user in container
- Scan images for vulnerabilities (e.g., `docker scan`)
- Use secrets for credentials instead of ENV vars
- Keep Docker and images updated

---

## 🏗️ Multi-stage Dockerfiles

Reduces image size by separating build and runtime stages.

```Dockerfile
FROM node:16 AS build
WORKDIR /app
COPY . .
RUN npm install && npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
```

---

## 🚀 Caching & Layering Optimizations

- Layers are cached, re-used unless changed
- Place least frequently changed commands (like `RUN npm install`) early
- Use `.dockerignore` to reduce build context
- Keep layers clean using `&&` and `rm`

```Dockerfile
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
```

---

## 🏢 Private Docker Registries

- Use Docker Hub or your own registry

```bash
docker login <registry-url>
docker tag myapp <registry-url>/myapp:tag
docker push <registry-url>/myapp:tag
docker pull <registry-url>/myapp:tag
```

- Self-hosted: `registry:2` image

```bash
docker run -d -p 5000:5000 --name registry registry:2
```

---
