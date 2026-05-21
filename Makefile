# Use Git Bash on Windows (make defaults to cmd.exe which breaks bash syntax)
SHELL := C:/Program Files/Git/bin/bash.exe

# =============================================================================
# MOVE-LIKE-BUFFET -- Makefile (idempotent -- safe to run any target at any time)
# =============================================================================
#
# IP STRATEGY:
#   Each VM instance gets its own static IP named buffett-ip-{INSTANCE}.
#   On first deploy the ephemeral IP GCP assigns is promoted to static.
#   On subsequent deploys the static IP already exists -- nothing changes.
#
# CHEAT SHEET:
#   make check        read-only -- see what's in GCP
#   make check-vm     read-only -- see VM + containers
#   make check-fw     read-only -- see firewall rules
#   make build        build Docker image locally
#   make up / down    local docker test
#   make push         tag + push image to Artifact Registry
#   make deploy       ensure VM + promote IP + start container
#   make all          build -> push -> deploy -> status
#   make redeploy     quick: rebuild + push + deploy
#   make status       check production health
#   make ssh          SSH into VM
#   make destroy      tear down EVERYTHING (VM, IP, firewall, AR images)
# =============================================================================


# -----------------------------------------------
# CONFIG -- loaded from deploy.env
# -----------------------------------------------

-include deploy.env

_check-config:
	@if [ -z "$(GCP_PROJECT)" ]; then \
		echo ""; \
		echo "ERROR: deploy.env not found or GCP_PROJECT not set."; \
		echo "Create deploy.env in project root with:"; \
		echo "  GCP_PROJECT=your-gcp-project-id"; \
		echo "  GCP_REGION=us-east1"; \
		echo "  AR_REPO=move-like-buffet"; \
		echo "  INSTANCE=buffett-prod-vm"; \
		echo "  ZONE=us-east1-c"; \
		echo "  MACHINE=e2-standard-2"; \
		echo "  VM_TAG=buffett-app"; \
		exit 1; \
	fi
	@echo "Config: project=$(GCP_PROJECT) instance=$(INSTANCE) zone=$(ZONE)"

# Derived
AR_HOST        = $(GCP_REGION)-docker.pkg.dev
AR_PREFIX      = $(AR_HOST)/$(GCP_PROJECT)/$(AR_REPO)
IMG_APP        = $(AR_PREFIX)/buffett-app
LOCAL_APP      = buffett-app
TIMESTAMP     := $(shell bash -c 'date +%Y%m%d-%H%M%S')
STATIC_IP_NAME = buffett-ip-$(INSTANCE)
FW_RULE_NAME   = $(VM_TAG)-allow-http

.PHONY: check check-vm check-fw build up down health auth push \
        deploy _ensure-vm _ensure-firewall _ensure-ar _wait-ssh _promote-ip \
        ssh status all redeploy clean _check-config destroy \
        _destroy-vm _destroy-ip _destroy-fw _destroy-ar


# =============================================================================
# ENSURE TARGETS -- idempotent infrastructure
# =============================================================================

_promote-ip: _check-config
	@STATIC_EXISTS=$$(gcloud compute addresses describe $(STATIC_IP_NAME) \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--format="get(address)" 2>/dev/null || echo ""); \
	if [ -n "$$STATIC_EXISTS" ]; then \
		echo "Static IP $(STATIC_IP_NAME) already exists ($$STATIC_EXISTS) -- nothing to do"; \
	else \
		CURRENT_IP=$$(gcloud compute instances describe $(INSTANCE) \
			--project=$(GCP_PROJECT) --zone=$(ZONE) \
			--format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo ""); \
		if [ -z "$$CURRENT_IP" ]; then \
			echo "ERROR: $(INSTANCE) has no external IP to promote."; \
			exit 1; \
		fi; \
		echo "Promoting ephemeral IP $$CURRENT_IP to static $(STATIC_IP_NAME)..."; \
		gcloud compute addresses create $(STATIC_IP_NAME) \
			--project=$(GCP_PROJECT) \
			--region=$(GCP_REGION) \
			--addresses=$$CURRENT_IP; \
		echo "IP $$CURRENT_IP is now static ($(STATIC_IP_NAME))"; \
	fi

_ensure-vm: _check-config _ensure-firewall
	@VM_STATUS=$$(gcloud compute instances describe $(INSTANCE) \
		--project=$(GCP_PROJECT) --zone=$(ZONE) \
		--format="get(status)" 2>/dev/null || echo "NOT_FOUND"); \
	echo "VM $(INSTANCE) status: $$VM_STATUS"; \
	if [ "$$VM_STATUS" = "NOT_FOUND" ]; then \
		echo "Creating VM $(INSTANCE)..."; \
		gcloud compute instances create $(INSTANCE) \
			--project=$(GCP_PROJECT) \
			--zone=$(ZONE) \
			--machine-type=$(MACHINE) \
			--boot-disk-size=30GB \
			--image-family=ubuntu-2204-lts \
			--image-project=ubuntu-os-cloud \
			--scopes=cloud-platform \
			--tags=$(VM_TAG) \
			--metadata=google-logging-enabled=true; \
		echo "VM created."; \
	elif [ "$$VM_STATUS" = "TERMINATED" ] || [ "$$VM_STATUS" = "STOPPED" ]; then \
		echo "Starting VM $(INSTANCE)..."; \
		gcloud compute instances start $(INSTANCE) \
			--project=$(GCP_PROJECT) --zone=$(ZONE); \
		echo "Waiting for VM to reach RUNNING..."; \
		for i in $$(seq 1 30); do \
			S=$$(gcloud compute instances describe $(INSTANCE) \
				--project=$(GCP_PROJECT) --zone=$(ZONE) \
				--format="get(status)" 2>/dev/null); \
			if [ "$$S" = "RUNNING" ]; then echo "  VM running ($${i}x2s)"; break; fi; \
			sleep 2; \
		done; \
	else \
		echo "VM $(INSTANCE) is $$VM_STATUS -- nothing to do"; \
	fi
	@$(MAKE) _wait-ssh
	@$(MAKE) _promote-ip

_wait-ssh: _check-config
	@echo "Waiting for SSH on $(INSTANCE)..."
	@for i in $$(seq 1 60); do \
		if gcloud compute ssh $(INSTANCE) \
			--project=$(GCP_PROJECT) --zone=$(ZONE) \
			--command="echo ok" 2>/dev/null | grep -q ok; then \
			echo "  SSH ready ($${i}x2s)"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "ERROR: SSH not reachable after 120s"; exit 1

_ensure-firewall: _check-config
	@if gcloud compute firewall-rules describe $(FW_RULE_NAME) \
		--project=$(GCP_PROJECT) > /dev/null 2>&1; then \
		echo "Firewall $(FW_RULE_NAME) -- exists"; \
	else \
		echo "Creating firewall rule $(FW_RULE_NAME) (port 8000)..."; \
		gcloud compute firewall-rules create $(FW_RULE_NAME) \
			--project=$(GCP_PROJECT) \
			--allow=tcp:8000 \
			--target-tags=$(VM_TAG) \
			--description="Buffett app HTTP access"; \
	fi

_ensure-ar: _check-config
	@if gcloud artifacts repositories describe $(AR_REPO) \
		--project=$(GCP_PROJECT) --location=$(GCP_REGION) > /dev/null 2>&1; then \
		echo "AR repo $(AR_REPO) -- exists"; \
	else \
		echo "Creating AR repo $(AR_REPO)..."; \
		gcloud artifacts repositories create $(AR_REPO) \
			--project=$(GCP_PROJECT) --repository-format=docker \
			--location=$(GCP_REGION); \
	fi


# =============================================================================
# CHECK -- read-only
# =============================================================================

check: _check-config
	@echo ""
	@echo "====== Project: $(GCP_PROJECT) | Region: $(GCP_REGION) ======"
	@echo "   AR Repo:  $(AR_REPO)"
	@echo "   VM:       $(INSTANCE) ($(ZONE))"
	@echo "   VM Tag:   $(VM_TAG)"
	@echo ""
	@echo "-- Artifact Registry --"
	@gcloud artifacts repositories list \
		--project=$(GCP_PROJECT) --location=$(GCP_REGION) \
		--format="table(name,format,sizeBytes)" 2>/dev/null
	@echo ""
	@echo "-- Images --"
	@gcloud artifacts docker images list $(AR_PREFIX) \
		--project=$(GCP_PROJECT) \
		--format="table(package,version,updateTime)" \
		--sort-by="~updateTime" 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "-- Static IPs --"
	@gcloud compute addresses list \
		--project=$(GCP_PROJECT) \
		--filter="region:$(GCP_REGION)" \
		--format="table(name,address,status,users)" 2>/dev/null || echo "  (none)"

check-vm: _check-config
	@echo "-- All VMs --"
	@gcloud compute instances list --project=$(GCP_PROJECT) \
		--format="table(name,zone,machineType,status,networkInterfaces[0].accessConfigs[0].natIP)"
	@echo ""
	@echo "-- Containers on $(INSTANCE) --"
	@gcloud compute ssh $(INSTANCE) --project=$(GCP_PROJECT) --zone=$(ZONE) \
		--command="sudo docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'" \
		2>/dev/null || echo "  (cannot reach $(INSTANCE))"

check-fw: _check-config
	@echo "-- Firewall rules --"
	@gcloud compute firewall-rules list --project=$(GCP_PROJECT) \
		--format="table(name,allowed,direction,targetTags)"


# =============================================================================
# BUILD
# =============================================================================

build:
	@echo "====== Building buffett-app ======"
	docker build -t $(LOCAL_APP):latest .
	@echo ""
	@docker images --filter "reference=$(LOCAL_APP)" \
		--format "  {{.Repository}}:{{.Tag}}  ({{.Size}})"


# =============================================================================
# LOCAL TEST
# =============================================================================

up:
	docker run -d --name buffett-app -p 8000:8000 --env-file backend/.env $(LOCAL_APP):latest
	@echo "Waiting for startup..."
	@for i in $$(seq 1 60); do \
		if curl -sf http://localhost:8000/health > /dev/null 2>&1; then \
			echo "  Ready ($${i}s) -- http://localhost:8000/index.html"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "  Did not start in 60s -- check: docker logs buffett-app"

down:
	docker stop buffett-app 2>/dev/null || true
	docker rm   buffett-app 2>/dev/null || true

health:
	@echo "-- Local health --"
	@curl -sf http://localhost:8000/health > /dev/null 2>&1 \
		&& echo "  App  http://localhost:8000/index.html  OK" \
		|| echo "  App  FAIL"


# =============================================================================
# PUSH -- tag + push to Artifact Registry
# =============================================================================

auth: _check-config
	gcloud auth configure-docker $(AR_HOST) --quiet

push: auth _ensure-ar
	@echo "====== Pushing to $(AR_PREFIX) ======"
	docker tag $(LOCAL_APP):latest $(IMG_APP):latest
	docker tag $(LOCAL_APP):latest $(IMG_APP):$(TIMESTAMP)
	docker push $(IMG_APP):latest
	docker push $(IMG_APP):$(TIMESTAMP)
	@echo "Pushed: latest + $(TIMESTAMP)"


# =============================================================================
# DEPLOY -- single container on VM
# =============================================================================

deploy: _ensure-vm
	@echo "====== Deploying buffett-app to $(INSTANCE) ======"
	$(eval VM_IP := $(shell gcloud compute instances describe $(INSTANCE) \
		--project=$(GCP_PROJECT) --zone=$(ZONE) \
		--format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null))
	@if [ -z "$(VM_IP)" ]; then \
		echo "ERROR: Cannot get external IP for $(INSTANCE)."; \
		exit 1; \
	fi
	@echo "  VM IP: $(VM_IP)"
	@echo '#!/bin/bash' > _deploy.sh
	@echo 'set -e' >> _deploy.sh
	@echo 'if ! command -v docker >/dev/null 2>&1; then echo "Installing Docker..."; curl -fsSL https://get.docker.com | sudo sh; fi' >> _deploy.sh
	@echo 'gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://$(AR_HOST)' >> _deploy.sh
	@echo 'sudo docker pull $(IMG_APP):latest' >> _deploy.sh
	@echo 'sudo docker stop buffett-app 2>/dev/null || true' >> _deploy.sh
	@echo 'sudo docker rm   buffett-app 2>/dev/null || true' >> _deploy.sh
	@echo 'sudo docker run -d \' >> _deploy.sh
	@echo '  --name buffett-app \' >> _deploy.sh
	@echo '  --restart unless-stopped \' >> _deploy.sh
	@echo '  -p 8000:8000 \' >> _deploy.sh
	@echo '  --env-file $$HOME/.env \' >> _deploy.sh
	@echo '  -e CORS_ORIGINS="*" \' >> _deploy.sh
	@echo '  $(IMG_APP):latest' >> _deploy.sh
	@echo 'echo "=== DEPLOYED ==="' >> _deploy.sh
	@echo 'sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"' >> _deploy.sh
	gcloud compute scp backend/.env $(INSTANCE):.env --project=$(GCP_PROJECT) --zone=$(ZONE)
	gcloud compute scp _deploy.sh $(INSTANCE):_deploy.sh --project=$(GCP_PROJECT) --zone=$(ZONE)
	-gcloud compute ssh $(INSTANCE) --project=$(GCP_PROJECT) --zone=$(ZONE) \
		--command="bash _deploy.sh && rm _deploy.sh"
	@rm -f _deploy.sh
	@echo ""
	@echo "====== LIVE ======"
	@echo "  App: http://$(VM_IP):8000/index.html"
	@echo "  API: http://$(VM_IP):8000/docs"


# =============================================================================
# STATUS
# =============================================================================

status: _check-config
	@VM_STATUS=$$(gcloud compute instances describe $(INSTANCE) \
		--project=$(GCP_PROJECT) --zone=$(ZONE) \
		--format="get(status)" 2>/dev/null || echo "NOT_FOUND"); \
	VM_IP=$$(gcloud compute instances describe $(INSTANCE) \
		--project=$(GCP_PROJECT) --zone=$(ZONE) \
		--format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo ""); \
	STATIC_IP=$$(gcloud compute addresses describe $(STATIC_IP_NAME) \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--format="get(address)" 2>/dev/null || echo "not promoted yet"); \
	echo ""; \
	echo "====== $(INSTANCE) | $$VM_STATUS | external=$$VM_IP | static=$(STATIC_IP_NAME)=$$STATIC_IP ======"; \
	echo ""; \
	if [ "$$VM_STATUS" = "RUNNING" ]; then \
		gcloud compute ssh $(INSTANCE) --project=$(GCP_PROJECT) --zone=$(ZONE) \
			--command="sudo docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'" \
			2>/dev/null || echo "  (SSH failed)"; \
		echo ""; \
		echo "-- Endpoints --"; \
		curl -sf http://$$VM_IP:8000/health > /dev/null 2>&1 \
			&& echo "  App       http://$$VM_IP:8000/index.html  OK" \
			|| echo "  App       FAIL (container may still be starting)"; \
	elif [ "$$VM_STATUS" = "NOT_FOUND" ]; then \
		echo "  VM does not exist. Run 'make deploy'."; \
	else \
		echo "  VM is $$VM_STATUS. Run 'make deploy'."; \
	fi


# =============================================================================
# SSH
# =============================================================================

ssh: _ensure-vm
	gcloud compute ssh $(INSTANCE) --project=$(GCP_PROJECT) --zone=$(ZONE)


# =============================================================================
# COMPOSITE
# =============================================================================

all: build push deploy status

redeploy: build push deploy status


# =============================================================================
# DESTROY -- tears down EVERYTHING this Makefile created
# =============================================================================

_destroy-vm: _check-config
	@VM_STATUS=$$(gcloud compute instances describe $(INSTANCE) \
		--project=$(GCP_PROJECT) --zone=$(ZONE) \
		--format="get(status)" 2>/dev/null || echo "NOT_FOUND"); \
	if [ "$$VM_STATUS" = "NOT_FOUND" ]; then \
		echo "VM $(INSTANCE) -- does not exist"; \
	else \
		echo "Deleting VM $(INSTANCE)..."; \
		gcloud compute instances delete $(INSTANCE) \
			--project=$(GCP_PROJECT) --zone=$(ZONE) --quiet; \
		echo "  VM deleted"; \
	fi

_destroy-ip: _check-config
	@STATIC_EXISTS=$$(gcloud compute addresses describe $(STATIC_IP_NAME) \
		--project=$(GCP_PROJECT) --region=$(GCP_REGION) \
		--format="get(address)" 2>/dev/null || echo ""); \
	if [ -z "$$STATIC_EXISTS" ]; then \
		echo "Static IP $(STATIC_IP_NAME) -- does not exist"; \
	else \
		echo "Releasing static IP $(STATIC_IP_NAME) ($$STATIC_EXISTS)..."; \
		gcloud compute addresses delete $(STATIC_IP_NAME) \
			--project=$(GCP_PROJECT) --region=$(GCP_REGION) --quiet; \
		echo "  IP released"; \
	fi

_destroy-fw: _check-config
	@if gcloud compute firewall-rules describe $(FW_RULE_NAME) \
		--project=$(GCP_PROJECT) > /dev/null 2>&1; then \
		echo "Deleting firewall rule $(FW_RULE_NAME)..."; \
		gcloud compute firewall-rules delete $(FW_RULE_NAME) \
			--project=$(GCP_PROJECT) --quiet; \
		echo "  Firewall rule deleted"; \
	else \
		echo "Firewall $(FW_RULE_NAME) -- does not exist"; \
	fi

_destroy-ar: _check-config
	@if gcloud artifacts repositories describe $(AR_REPO) \
		--project=$(GCP_PROJECT) --location=$(GCP_REGION) > /dev/null 2>&1; then \
		echo "Deleting AR repo $(AR_REPO) and all images..."; \
		gcloud artifacts repositories delete $(AR_REPO) \
			--project=$(GCP_PROJECT) --location=$(GCP_REGION) --quiet; \
		echo "  AR repo deleted"; \
	else \
		echo "AR repo $(AR_REPO) -- does not exist"; \
	fi

destroy: _check-config
	@echo ""
	@echo "====== DESTROYING ALL RESOURCES FOR $(INSTANCE) ======"
	@echo "  This will delete:"
	@echo "    - VM:       $(INSTANCE)"
	@echo "    - Static IP: $(STATIC_IP_NAME)"
	@echo "    - Firewall:  $(FW_RULE_NAME)"
	@echo "    - AR Repo:   $(AR_REPO) (all images)"
	@echo ""
	@read -p "  Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" != "yes" ]; then \
		echo "  Aborted."; \
		exit 1; \
	fi
	@$(MAKE) _destroy-vm
	@$(MAKE) _destroy-ip
	@$(MAKE) _destroy-fw
	@$(MAKE) _destroy-ar
	@echo ""
	@echo "====== ALL RESOURCES DESTROYED ======"


# =============================================================================
# CLEAN -- local only
# =============================================================================

clean:
	rm -f _deploy.sh
	docker stop buffett-app 2>/dev/null || true
	docker rm   buffett-app 2>/dev/null || true