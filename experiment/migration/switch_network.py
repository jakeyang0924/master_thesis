import subprocess

class Route:
    @classmethod
    def get_default_routes(cls):
        # run the ip command to get all default routes
        command = ["ip", "route", "show", "default"]
        output = subprocess.check_output(command).decode().strip()

        # split the output into individual routes and parse each one
        routes = []
        for line in output.split("\n"):
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] != "default":
                continue
            gateway_ip = parts[2]
            device_name = parts[4] if len(parts) > 4 else ""
            metric = None
            proto = None
            for i in range(len(parts)):
                if parts[i] == "metric":
                    metric = parts[i+1]
                elif parts[i] == "proto":
                    proto = parts[i+1]
            routes.append((gateway_ip, device_name, metric, proto))

        return routes
    
    @classmethod
    def add_default_route(cls, gateway_ip, device_name, metric=100):
        # run the ip command to add the default route
        command = ["sudo", "ip", "route", "add", "default", "via", gateway_ip, "dev", device_name, "metric", str(metric)]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, input=b"nems@704\n")
            return True
        except subprocess.CalledProcessError:
            return False
        
        
    
    @classmethod
    def delete_default_route(cls, gateway_ip, device_name):
        # construct the ip command to delete the default route
        command = ["sudo", "ip", "route", "del", "default", "via", gateway_ip, "dev", device_name]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, input=b"nems@704\n")
            return True
        except subprocess.CalledProcessError:
            return False

    @classmethod
    def switch_default_route(cls):
        # get all default routes
        routes = cls.get_default_routes()

        # if there are two default routes, adjust the metrics
        if len(routes) == 2:
            # find the route with the larger metric
            larger_route = max(routes, key=lambda route: int(route[2]))
            smaller_route = min(routes, key=lambda route: int(route[2]))

            # reset both metrics to a larger value if they are 0
            flag = False
            if int(smaller_route[2]) == 1:
                flag = True
                smaller_metric = 100
            else:
                smaller_metric = int(smaller_route[2])
            new_metric = smaller_metric - 1

            if flag:
                if not (cls.delete_default_route(smaller_route[0], smaller_route[1]) and \
                    cls.add_default_route(smaller_route[0], smaller_route[1], smaller_metric)):
                        return False
            if cls.delete_default_route(larger_route[0], larger_route[1]) and \
                cls.add_default_route(larger_route[0], larger_route[1], new_metric):
                return True

        return False

    @classmethod
    def get_current_route(cls):
        routes = cls.get_default_routes()
        current_route = min(routes, key=lambda route: int(route[2]))
        current_route_dict = {
            'ip': current_route[0],
            'dev': current_route[1],
            'metric': current_route[2],
            'proto': current_route[3]
        }
        return current_route_dict


Route.switch_default_route()