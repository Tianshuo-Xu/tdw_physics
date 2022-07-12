# bash upload.sh 1
# bash upload.sh render1

re='^[0-9]+$'
if [[ $1 =~ $re ]]
then
    echo "upload to node"$1
    rsync -vaP -e "ssh" --exclude=log/ --exclude=log2 --exclude=data --exclude=dump * hsiaoyut@node$1-ccncluster.stanford.edu:~/2021/tdw_physics/ #dev_head_compute0#
else
    echo "upload to render1"
    rsync -vaP -e "ssh" --exclude=log/ --exclude=log2 --exclude=data --exclude=dump  * hsiaoyut@render1-neuroaicluster.stanford.edu:~/2021/tdw_physics/ #dev_head_compute0#
fi


