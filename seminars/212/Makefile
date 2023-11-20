build:
	docker build \
	-t "hse-dl-212:latest" \
	-f Dockerfile \
	.

run:
	docker run \
	--gpus all \
	--network=host \
	--ipc=host \
	--mount type=bind,source=/home/$$(whoami)/PycharmProjects/intro-to-dl-hse/seminars/212/,target=/srv/www/hse-dl/ \
	hse-dl-212:latest

run-bash:
	docker run -it \
	--gpus all \
	--network=host \
	--ipc=host \
	--mount type=bind,source=/home/$$(whoami)/PycharmProjects/intro-to-dl-hse/seminars/212/,target=/srv/www/hse-dl/ \
	hse-dl-212:latest \
	bash
