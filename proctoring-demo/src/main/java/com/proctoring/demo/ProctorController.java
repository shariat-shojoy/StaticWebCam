package com.proctoring.demo;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
public class ProctorController {
    @PostMapping("/violation")
    public ResponseEntity<String> violation(
            @RequestBody Map<String,String> body){

        System.out.println(
                "VIOLATION: "
                        + body.get("type")
        );

        return ResponseEntity.ok(
                "Logged"
        );
    }

    @PostMapping("/upload")
    public Map<String, Object> upload(
            @RequestParam("file") MultipartFile file) throws Exception {

        RestTemplate restTemplate = new RestTemplate();

        ByteArrayResource resource =
                new ByteArrayResource(file.getBytes()) {
                    @Override
                    public String getFilename() {
                        return file.getOriginalFilename();
                    }
                };

        MultiValueMap<String, Object> body =
                new LinkedMultiValueMap<>();

        body.add("file", resource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> request =
                new HttpEntity<>(body, headers);

        ResponseEntity<Map> response =
                restTemplate.postForEntity(
                        "http://127.0.0.1:8000/analyze",
                        request,
                        Map.class
                );

        return response.getBody();
    }
}