# API Documentation

Base URL: `http://localhost:8000`

## `POST /auth/register`

Creates a user.

Request:

```json
{"email":"user@example.com","password":"minimum-8-chars"}
```

## `POST /auth/token`

OAuth2 password form login. The username field is the user email.

## `POST /upload`

Multipart upload field: `file`.

Response:

```json
{"job_id":"...","status":"uploaded"}
```

## `POST /predict`

Request:

```json
{"job_id":"..."}
```

Response:

```json
{
  "job_id": "...",
  "status": "completed",
  "cloud_mask_url": "/download/<job>?asset=mask",
  "download_url": "/download/<job>?asset=output",
  "metrics": {"psnr": 28.2},
  "confidence_score": 0.81
}
```

## `GET /download/{job_id}`

Query parameter `asset` can be `input`, `mask`, or `output`.

## `GET /history`

Returns recent inference jobs.
