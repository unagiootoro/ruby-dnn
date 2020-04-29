require "sinatra"
require "sinatra/reloader"
require "json"
require "base64"
require_relative "mnist_predict"

get "/" do
  erb :index
end

post "/predict" do
  json = request.body.read
  params = JSON.parse(json, symbolize_names: true)
  img = Base64.decode64(params[:img])
  width = params[:width].to_i
  height = params[:height].to_i
  result = mnist_predict(img, width, height)
  JSON.dump(result)
end
