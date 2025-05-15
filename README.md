# LLM Portal

## Configuration Requirements for Deployment

To deploy the LLM Portal, you need to set up a few configurations. Below are the steps to configure your environment.

### 1. Database configuration 

The database settings are defined in the `.configs/database.yaml` file.

**Required Fields**

- `type`: The type of database you are using. Supported types include `sqlite`, `postgresql`, and `mysql`.
- `framework`: Just support `sqlalchemy` for now.
- `connection`: The connection settings for your database.
    - `url`: The URL to connect to your database. For SQLite, it should be in the format `sqlite:///path/to/database.db`. For PostgreSQL and MySQL, it should be in the format `postgresql://user:password@host:port/database` or `mysql://user:password@host:port/database`.
     
**Example Configuration**

```yaml
database:
  type: sqlite
  framework: sqlalchemy
  connection:
    url: sqlite:///./llm_portal.sqlite
```

### 2. Google Vertex AI configuration

The Vertex AI settings are defined in the `.configs/google.yaml` file.

**Required Fields**
- `project_id`: The ID of your Google Cloud project.
- `project_location`: The location of your Google Cloud resources, currently only `us-central1` is supported for Vertex AI.
- `credentials_path`: The path to your Google Cloud credentials file. This file is used to authenticate your application with Google Cloud services.
- 
**Example Configuration**

```yaml
vertexai:
  project_id: gen-lang-client-0884898390
  project_location: us-central1
  credentials_path: /home/delus/.config/gcloud/application_default_credentials.json
```
For Google Vertex AI, you need to set up a Google Cloud project and enable the Vertex AI API. 
Here is the link to get Google Vertex AI credentials: [https://cloud.google.com/vertex-ai/docs/authentication](https://cloud.google.com/vertex-ai/docs/authentication)

Below is some example command to log in and get the credentials with Google Cloud CLI:
```bash
Run command: 
```bash
gcloud init
gcloud auth application-default login
```
and change the `credentials_path` in `.config/google.yaml` for your `credentials.json` file

### 3. Message Broker configuration
The message broker settings are defined in the `.configs/message_broker.yaml` file.

**Configuration Fields**
- `framework`: The message broker framework to use (e.g., redis).
- `connection`:
  - `host`: The hostname or IP address of the message broker server.
  - `port`: The port number for the message broker server (default for Redis is 6379).

**Example Configuration**

```yaml
message_broker:
  framework: redis
  connection:
    host: 192.168.1.100
    port: 6379
```

Currently message broker is not implemented yet, but you can set up the configuration for future use.