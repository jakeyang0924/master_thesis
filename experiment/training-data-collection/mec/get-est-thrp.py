import redis
r = redis.Redis(host='localhost', port=6379, db=0)

est_5g = r.hget('throughput', '5g').decode('utf-8')
est_wifi = r.hget('throughput', 'wifi').decode('utf-8')

print(f'{est_5g} {est_wifi}')