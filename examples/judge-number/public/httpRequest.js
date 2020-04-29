class HttpRequest {
    static get(path, responseCallback) {
        const req = new HttpRequest(path, "GET", responseCallback);
        req.send();
        return req;
    }

    static post(path, params, responseCallback) {
        const req = new HttpRequest(path, "POST", responseCallback);
        req.send(params);
        return req;
    }

    constructor(path, method, responseCallback) {
        this._path = path;
        this._method = method;
        this._responseCallback = responseCallback;
    }

    send(params = null) {
        const xhr = new XMLHttpRequest();
        xhr.open(this._method, this._path);
        let json = null;
        if (params) json = JSON.stringify(params);
        xhr.addEventListener("load", (e) => {
            const res = {
                response: xhr.response,
                event: e
            };
            this._responseCallback(res);
        });
        xhr.send(json);
    }
}

class Base64 {
    static encode(obj) {
        if (typeof(obj) === "string") {
            return btoa(obj);
        } else if (obj instanceof Uint8Array || obj instanceof Uint8ClampedArray) {
            return btoa(String.fromCharCode(...obj));
        }
    }
}
