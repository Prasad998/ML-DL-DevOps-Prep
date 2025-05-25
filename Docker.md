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
---

# 🧪 Docker Interview Questions 

> ✅ 30 Most Common Questions with Answers

---

### 1. **What is Docker and why is it used?**
Docker is a containerization platform used to package applications and their dependencies into isolated units called containers. It ensures consistency across environments and simplifies deployment.

---

### 2. **What is the difference between a container and a virtual machine (VM)?**
- **Containers**: Lightweight, share the host OS kernel, start faster.
- **VMs**: Heavier, each runs a full OS, more resource intensive.

---

### 3. **What is a Dockerfile?**
A Dockerfile is a script of instructions on how to build a Docker image. It defines base image, working directory, commands to run, etc.

---

### 4. **How do you build and run a Docker image?**
```bash
docker build -t myimage .
docker run -p 3000:3000 myimage
```

---

### 5. **What is the difference between CMD and ENTRYPOINT in Dockerfile?**
- `CMD`: Provides default arguments to the container.
- `ENTRYPOINT`: Specifies the executable; CMD becomes default parameters for ENTRYPOINT.

---

### 6. **How can you persist data in Docker?**
Use **volumes**:
```bash
docker volume create mydata
docker run -v mydata:/app/data myimage
```

---

### 7. **What is Docker Compose?**
Tool for defining and running multi-container applications using a `docker-compose.yml` file.

---

### 8. **What is the difference between COPY and ADD in Dockerfile?**
- `COPY`: Basic file copy.
- `ADD`: Can also extract archives and support URLs.

---

### 9. **What is the difference between a Docker image and a container?**
- **Image**: Blueprint or snapshot.
- **Container**: Running instance of an image.

---

### 10. **What are some best practices for writing Dockerfiles?**
- Use minimal base images.
- Leverage multi-stage builds.
- Order instructions to maximize caching.
- Clean up unnecessary files.

---

### 11. **How do you check running containers?**
```bash
docker ps
```

---

### 12. **How to expose and bind ports?**
```bash
docker run -p 8080:80 myapp
```
Maps host port 8080 to container port 80.

---

### 13. **How do you remove all stopped containers and unused images?**
```bash
docker system prune
```

---

### 14. **What is Docker Hub?**
A public registry to store and share Docker images.

---

### 15. **What are Docker volumes?**
Persistent storage independent of containers.

---

### 16. **How do containers communicate with each other?**
Via Docker **bridge network** or a custom network.

---

### 17. **What is the default network driver in Docker?**
**bridge**

---

### 18. **What is the purpose of `.dockerignore`?**
To exclude files from the build context (like `.git`, `node_modules`).

---

### 19. **How do you run a container in background?**
```bash
docker run -d <image>
```

---

### 20. **What is the difference between `RUN`, `CMD`, and `ENTRYPOINT`?**
- `RUN`: Executes at build time.
- `CMD`: Sets default command.
- `ENTRYPOINT`: Defines fixed executable.

---

### 21. **What is multi-stage build in Docker?**
Optimize final image size by separating build and runtime environments.

---

### 22. **How do you inspect a container or image?**
```bash
docker inspect <container|image>
```

---

### 23. **How to attach to a running container?**
```bash
docker exec -it <container-id> bash
```

---

### 24. **How do you define environment variables in Docker?**
```Dockerfile
ENV PORT=3000
```

---

### 25. **How do you scale containers using Docker Compose?**
```bash
docker-compose up --scale web=3
```

---

### 26. **What is the purpose of health checks in Docker?**
Used to monitor container status using the `HEALTHCHECK` directive.

---

### 27. **What’s the difference between `docker stop` and `docker kill`?**
- `stop`: Graceful shutdown.
- `kill`: Immediate termination.

---

### 28. **What is a dangling image?**
An image without a tag or not referenced by any container.

---

### 29. **What is overlay network in Docker?**
Used for multi-host networking, especially in Docker Swarm.

---

### 30. **How can you secure a Docker container?**
- Use least-privileged user.
- Scan images for vulnerabilities.
- Use read-only filesystem.

---
