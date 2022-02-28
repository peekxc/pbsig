# Create an SDK manager                                                                                           
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, "")                                                                                            
importer = fbx.FbxImporter.Create(manager, "")
milfalcon = "samples/millenium-falcon/millenium-falcon.fbx"                                                                       
importstat = importer.Initialize(milfalcon, -1)
importstat = importer.Import(scene)                                                                              
exporter = fbx.FbxExporter.Create(manager, "")

save_path = "samples/millenium-falcon/millenium-falcon.obj"

# Specify the path and name of the file to be imported                                                                            
exportstat = exporter.Initialize(save_path, -1)

exportstat = exporter.Export(scene)